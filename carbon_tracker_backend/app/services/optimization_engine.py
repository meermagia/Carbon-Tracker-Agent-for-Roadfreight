"""
Route optimization engine (OR-Tools).

Recommends routing strategies that reduce carbon emissions while maintaining
cost efficiency by minimizing a weighted objective:

    alpha * Cost + beta * CarbonEmissions

This module integrates with:
- `carbon_engine` via vehicle emission profiles and the same emissions formula
- `graph_builder` by consuming the NetworkX logistics graph to generate candidate paths

Design notes:
- The core optimization is a mixed-integer program (MIP) solved with OR-Tools CBC.
- Each shipment is assigned to exactly one candidate route (path).
- Consolidation is modeled via integer trip counts per route with capacity constraints.
  This enables the optimizer to suggest consolidation opportunities (fewer trips for
  the same demand). Optional per-trip overhead cost/emissions can be configured to
  make consolidation yield tangible reductions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from ortools.linear_solver import pywraplp

from app.models.shipment_model import Shipment
from app.utils.emission_factors import VehicleEmissionProfile, get_vehicle_profile
from app.services.graph_builder import ATTR_CARBON_INTENSITY, ATTR_DISTANCE


@dataclass(frozen=True)
class RouteCandidate:
    """
    A candidate route option for a shipment.

    Attributes:
        route_id: Stable identifier within an optimization run.
        path: Ordered list of node labels representing the path.
        distance_km: Total route distance in km.
    """

    route_id: str
    path: List[str]
    distance_km: float


@dataclass(frozen=True)
class ShipmentOptimizationRecord:
    """
    Minimal shipment record used by the optimizer.
    """

    shipment_id: str
    origin: str
    destination: str
    distance_km: float
    weight_tons: float
    vehicle_type: str
    baseline_cost: float
    baseline_emissions_kg: float
    vehicle_profile: VehicleEmissionProfile


@dataclass(frozen=True)
class OptimizationInput:
    """
    Prepared input for the optimization run.

    Attributes:
        shipments: List of shipment records.
        candidates_by_shipment: For each shipment index i, a list of route candidates.
        alpha: Weight for cost.
        beta: Weight for emissions.
        cost_per_km: Linear variable cost per km per shipment.
        per_trip_overhead_cost: Optional overhead cost per trip (encourages consolidation).
        per_trip_overhead_emissions_kg: Optional overhead emissions per trip (encourages consolidation).
    """

    shipments: List[ShipmentOptimizationRecord]
    candidates_by_shipment: List[List[RouteCandidate]]
    alpha: float
    beta: float
    cost_per_km: float
    per_trip_overhead_cost: float
    per_trip_overhead_emissions_kg: float


@dataclass(frozen=True)
class OptimizationSolution:
    """
    Solver output.
    """

    shipment_to_candidate_idx: Dict[int, int]
    trips_by_candidate: Dict[str, int]
    objective_value: float
    optimized_cost: float
    optimized_emissions_kg: float


def _path_distance_km(graph: nx.DiGraph, path: Sequence[str]) -> float:
    """
    Sum edge distances along a path using `graph_builder.ATTR_DISTANCE`.
    """
    total = 0.0
    for u, v in zip(path, path[1:], strict=False):
        edge = graph.edges[u, v]
        total += float(edge.get(ATTR_DISTANCE, 0.0))
    return total


def _find_candidate_paths(
    graph: nx.DiGraph,
    origin: str,
    destination: str,
    *,
    max_candidates: int = 2,
) -> List[List[str]]:
    """
    Generate a small set of candidate paths for a shipment.

    Strategy:
    - shortest path by distance
    - shortest path by carbon-weighted distance (intensity * distance)

    Deduplicates paths and returns up to `max_candidates` paths.
    """
    if origin not in graph or destination not in graph:
        return []

    paths: List[List[str]] = []

    # Candidate 1: shortest by distance
    try:
        p_dist = nx.shortest_path(graph, origin, destination, weight=ATTR_DISTANCE)
        paths.append(list(p_dist))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # Candidate 2: shortest by carbon intensity * distance
    def carbon_weight(u: str, v: str, d: dict) -> float:
        dist = float(d.get(ATTR_DISTANCE, 0.0))
        intensity = float(d.get(ATTR_CARBON_INTENSITY, 0.0))
        # If intensity is missing, default to 0.0 so distance dominates.
        return max(dist * max(intensity, 0.0), 0.0)

    try:
        p_carbon = nx.shortest_path(graph, origin, destination, weight=carbon_weight)
        paths.append(list(p_carbon))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # Deduplicate while preserving order
    uniq: List[List[str]] = []
    seen: set[tuple[str, ...]] = set()
    for p in paths:
        key = tuple(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)

    return uniq[:max_candidates]


def _estimate_emissions_kg(distance_km: float, weight_tons: float, profile: VehicleEmissionProfile) -> float:
    """
    Estimate shipment emissions using the same formula as `carbon_engine`:

        CO2 = Distance × EmissionFactor(vehicle_type) × (shipment_weight / vehicle_capacity)
    """
    load_adjustment = weight_tons / profile.capacity_tons
    return distance_km * profile.emission_factor_kg_co2e_per_km * load_adjustment


def prepare_optimization_inputs(
    shipments: Iterable[Shipment],
    *,
    graph: Optional[nx.DiGraph] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    cost_per_km: float = 1.0,
    max_route_candidates: int = 2,
    per_trip_overhead_cost: float = 0.0,
    per_trip_overhead_emissions_kg: float = 0.0,
) -> OptimizationInput:
    """
    Prepare optimization inputs from shipment records and an optional logistics graph.

    If a graph is provided, candidate paths are derived from it. Otherwise, only the
    direct origin->destination route is used.

    Args:
        shipments: Iterable of Shipment ORM entities.
        graph: Optional NetworkX logistics graph (from `graph_builder`).
        alpha: Cost weight in objective.
        beta: Emissions weight in objective.
        cost_per_km: Linear cost per km for each shipment route assignment.
        max_route_candidates: Max candidate routes per shipment (when graph is provided).
        per_trip_overhead_cost: Optional fixed overhead cost per trip (supports consolidation modeling).
        per_trip_overhead_emissions_kg: Optional fixed overhead emissions per trip.

    Returns:
        OptimizationInput ready for the OR-Tools solver.
    """
    shipment_records: List[ShipmentOptimizationRecord] = []
    candidates_by_shipment: List[List[RouteCandidate]] = []

    for idx, s in enumerate(shipments):
        profile = get_vehicle_profile(s.transport_mode)

        baseline_emissions = float(s.co2e_kg) if s.co2e_kg is not None else _estimate_emissions_kg(
            s.distance_km, s.weight_tons, profile
        )
        baseline_cost = float(s.distance_km) * float(cost_per_km)

        shipment_records.append(
            ShipmentOptimizationRecord(
                shipment_id=s.shipment_id,
                origin=s.origin_location,
                destination=s.destination_location,
                distance_km=float(s.distance_km),
                weight_tons=float(s.weight_tons),
                vehicle_type=s.transport_mode,
                baseline_cost=baseline_cost,
                baseline_emissions_kg=baseline_emissions,
                vehicle_profile=profile,
            )
        )

        # Candidate route generation
        if graph is None:
            candidates_by_shipment.append(
                [
                    RouteCandidate(
                        route_id=f"s{idx}_direct",
                        path=[s.origin_location, s.destination_location],
                        distance_km=float(s.distance_km),
                    )
                ]
            )
            continue

        paths = _find_candidate_paths(
            graph,
            s.origin_location,
            s.destination_location,
            max_candidates=max_route_candidates,
        )
        if not paths:
            # Fall back to direct distance if graph has no path
            candidates_by_shipment.append(
                [
                    RouteCandidate(
                        route_id=f"s{idx}_direct_fallback",
                        path=[s.origin_location, s.destination_location],
                        distance_km=float(s.distance_km),
                    )
                ]
            )
            continue

        candidates: List[RouteCandidate] = []
        for j, p in enumerate(paths):
            dist = _path_distance_km(graph, p)
            candidates.append(RouteCandidate(route_id=f"s{idx}_p{j}", path=p, distance_km=float(dist)))
        candidates_by_shipment.append(candidates)

    return OptimizationInput(
        shipments=shipment_records,
        candidates_by_shipment=candidates_by_shipment,
        alpha=float(alpha),
        beta=float(beta),
        cost_per_km=float(cost_per_km),
        per_trip_overhead_cost=float(per_trip_overhead_cost),
        per_trip_overhead_emissions_kg=float(per_trip_overhead_emissions_kg),
    )


def run_ortools_solver(opt_input: OptimizationInput) -> OptimizationSolution:
    """
    Run the OR-Tools MIP solver to find optimized route assignments.

    Model:
    - Each shipment chooses exactly 1 candidate route.
    - Capacity-constrained consolidation is modeled via integer trip counts per candidate route.

    Objective:
        alpha * Cost + beta * Emissions
    """
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver could not be created")

    shipments = opt_input.shipments
    candidates_by_shipment = opt_input.candidates_by_shipment

    # Decision variables x[i,j] for shipment i choosing candidate j
    x: Dict[tuple[int, int], pywraplp.Variable] = {}

    # trips per candidate route option (keyed by RouteCandidate.route_id)
    trips: Dict[str, pywraplp.Variable] = {}

    # Build variables and assignment constraints
    for i, cand_list in enumerate(candidates_by_shipment):
        ct = solver.Constraint(1.0, 1.0, f"assign_{i}")
        for j, cand in enumerate(cand_list):
            var = solver.IntVar(0.0, 1.0, f"x_{i}_{j}")
            x[(i, j)] = var
            ct.SetCoefficient(var, 1.0)

            if cand.route_id not in trips:
                trips[cand.route_id] = solver.IntVar(0.0, solver.infinity(), f"trips_{cand.route_id}")

    # Capacity constraints: sum(weight_i * x[i,j]) <= capacity * trips[candidate]
    for i, cand_list in enumerate(candidates_by_shipment):
        w = shipments[i].weight_tons
        cap = shipments[i].vehicle_profile.capacity_tons
        for j, cand in enumerate(cand_list):
            # We need one constraint per candidate aggregating across shipments that select it.
            # Build later in an aggregation loop; here we just collect coefficients.
            _ = (w, cap, cand.route_id)

    # Aggregate by candidate route_id
    candidate_to_terms: Dict[str, List[tuple[pywraplp.Variable, float, float]]] = {}
    for i, cand_list in enumerate(candidates_by_shipment):
        w = shipments[i].weight_tons
        cap = shipments[i].vehicle_profile.capacity_tons
        for j, cand in enumerate(cand_list):
            candidate_to_terms.setdefault(cand.route_id, []).append((x[(i, j)], w, cap))

    for route_id, terms in candidate_to_terms.items():
        # Use the minimum capacity among shipments that could take this route_id.
        # In practice, route candidates are shipment-specific; this keeps constraints safe.
        cap_min = min(cap for _, _, cap in terms) if terms else 0.0
        ct = solver.Constraint(0.0, solver.infinity(), f"cap_{route_id}")
        # sum(w * x) - cap_min * trips <= 0
        for var, w, _cap in terms:
            ct.SetCoefficient(var, float(w))
        ct.SetCoefficient(trips[route_id], -float(cap_min))

    # Objective
    objective = solver.Objective()
    objective.SetMinimization()

    alpha = opt_input.alpha
    beta = opt_input.beta

    # Linear per-shipment cost/emissions
    for i, cand_list in enumerate(candidates_by_shipment):
        srec = shipments[i]
        profile = srec.vehicle_profile
        for j, cand in enumerate(cand_list):
            distance = cand.distance_km
            cost = distance * opt_input.cost_per_km
            emissions = _estimate_emissions_kg(distance, srec.weight_tons, profile)

            objective.SetCoefficient(x[(i, j)], alpha * cost + beta * emissions)

    # Optional per-trip overhead (encourages consolidation if > 0)
    if opt_input.per_trip_overhead_cost != 0.0 or opt_input.per_trip_overhead_emissions_kg != 0.0:
        for route_id, tvar in trips.items():
            overhead = alpha * opt_input.per_trip_overhead_cost + beta * opt_input.per_trip_overhead_emissions_kg
            objective.SetCoefficient(tvar, float(overhead))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Optimization did not converge to optimal. status={status}")

    shipment_to_candidate_idx: Dict[int, int] = {}
    for i, cand_list in enumerate(candidates_by_shipment):
        chosen_j = None
        for j in range(len(cand_list)):
            if x[(i, j)].solution_value() >= 0.5:
                chosen_j = j
                break
        if chosen_j is None:
            # Should not happen due to equality constraint
            chosen_j = 0
        shipment_to_candidate_idx[i] = chosen_j

    trips_by_candidate: Dict[str, int] = {
        rid: int(round(var.solution_value())) for rid, var in trips.items()
    }

    # Compute optimized totals
    optimized_cost = 0.0
    optimized_emissions = 0.0
    for i, j in shipment_to_candidate_idx.items():
        srec = shipments[i]
        cand = candidates_by_shipment[i][j]
        optimized_cost += cand.distance_km * opt_input.cost_per_km
        optimized_emissions += _estimate_emissions_kg(
            cand.distance_km, srec.weight_tons, srec.vehicle_profile
        )

    # Add overheads
    if opt_input.per_trip_overhead_cost != 0.0:
        optimized_cost += sum(trips_by_candidate.values()) * opt_input.per_trip_overhead_cost
    if opt_input.per_trip_overhead_emissions_kg != 0.0:
        optimized_emissions += sum(trips_by_candidate.values()) * opt_input.per_trip_overhead_emissions_kg

    return OptimizationSolution(
        shipment_to_candidate_idx=shipment_to_candidate_idx,
        trips_by_candidate=trips_by_candidate,
        objective_value=solver.Objective().Value(),
        optimized_cost=optimized_cost,
        optimized_emissions_kg=optimized_emissions,
    )


def get_optimized_routes_and_reductions(
    opt_input: OptimizationInput,
    solution: OptimizationSolution,
) -> Dict[str, object]:
    """
    Build an integration-friendly result object with:
    - optimized route assignments
    - consolidation opportunities
    - estimated emission reduction vs baseline
    """
    shipments = opt_input.shipments
    candidates_by_shipment = opt_input.candidates_by_shipment

    baseline_cost = sum(s.baseline_cost for s in shipments)
    baseline_emissions = sum(s.baseline_emissions_kg for s in shipments)

    optimized_assignments: List[Dict[str, object]] = []
    shipments_per_candidate: Dict[str, int] = {}
    weight_per_candidate: Dict[str, float] = {}

    for i, srec in enumerate(shipments):
        chosen_j = solution.shipment_to_candidate_idx[i]
        cand = candidates_by_shipment[i][chosen_j]

        est_cost = cand.distance_km * opt_input.cost_per_km
        est_emissions = _estimate_emissions_kg(cand.distance_km, srec.weight_tons, srec.vehicle_profile)

        shipments_per_candidate[cand.route_id] = shipments_per_candidate.get(cand.route_id, 0) + 1
        weight_per_candidate[cand.route_id] = weight_per_candidate.get(cand.route_id, 0.0) + srec.weight_tons

        optimized_assignments.append(
            {
                "shipment_id": srec.shipment_id,
                "origin": srec.origin,
                "destination": srec.destination,
                "vehicle_type": srec.vehicle_type,
                "weight_tons": srec.weight_tons,
                "chosen_route_id": cand.route_id,
                "path": cand.path,
                "distance_km": cand.distance_km,
                "estimated_cost": est_cost,
                "estimated_emissions_kg": est_emissions,
            }
        )

    # Consolidation opportunities:
    # - Baseline assumption: 1 shipment == 1 trip
    # - Optimized: trips per chosen candidate route_id
    consolidation: List[Dict[str, object]] = []
    for route_id, assigned_count in shipments_per_candidate.items():
        trips = solution.trips_by_candidate.get(route_id, 0)
        total_weight = weight_per_candidate.get(route_id, 0.0)
        if assigned_count <= 0:
            continue
        if trips <= 0:
            # If overhead is 0, solver can keep trips at 0 while satisfying capacity constraint
            # only if all assigned weights are 0. Guard here for reporting.
            trips = max(1, ceil(total_weight / max(shipments[0].vehicle_profile.capacity_tons, 1e-9)))

        if trips < assigned_count:
            consolidation.append(
                {
                    "route_id": route_id,
                    "assigned_shipments": assigned_count,
                    "recommended_trips": trips,
                    "estimated_shipments_consolidated": assigned_count - trips,
                    "total_weight_tons": total_weight,
                }
            )

    optimized_cost = float(solution.optimized_cost)
    optimized_emissions = float(solution.optimized_emissions_kg)

    emission_reduction = baseline_emissions - optimized_emissions
    emission_reduction_pct = (emission_reduction / baseline_emissions * 100.0) if baseline_emissions > 0 else 0.0

    return {
        "objective": {
            "alpha": opt_input.alpha,
            "beta": opt_input.beta,
            "value": float(solution.objective_value),
        },
        "totals": {
            "baseline_cost": float(baseline_cost),
            "baseline_emissions_kg": float(baseline_emissions),
            "optimized_cost": optimized_cost,
            "optimized_emissions_kg": optimized_emissions,
            "emission_reduction_kg": float(emission_reduction),
            "emission_reduction_pct": float(emission_reduction_pct),
        },
        "optimized_assignments": optimized_assignments,
        "consolidation_opportunities": consolidation,
        "trips_by_route": dict(solution.trips_by_candidate),
    }


class OptimizationEngine:
    """
    Service wrapper around the module-level functions.
    """

    def prepare_optimization_inputs(
        self,
        shipments: Iterable[Shipment],
        *,
        graph: Optional[nx.DiGraph] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        cost_per_km: float = 1.0,
        max_route_candidates: int = 2,
        per_trip_overhead_cost: float = 0.0,
        per_trip_overhead_emissions_kg: float = 0.0,
    ) -> OptimizationInput:
        return prepare_optimization_inputs(
            shipments,
            graph=graph,
            alpha=alpha,
            beta=beta,
            cost_per_km=cost_per_km,
            max_route_candidates=max_route_candidates,
            per_trip_overhead_cost=per_trip_overhead_cost,
            per_trip_overhead_emissions_kg=per_trip_overhead_emissions_kg,
        )

    def run_solver(self, opt_input: OptimizationInput) -> OptimizationSolution:
        return run_ortools_solver(opt_input)

    def optimize(
        self,
        shipments: Iterable[Shipment],
        *,
        graph: Optional[nx.DiGraph] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        cost_per_km: float = 1.0,
        max_route_candidates: int = 2,
        per_trip_overhead_cost: float = 0.0,
        per_trip_overhead_emissions_kg: float = 0.0,
    ) -> Dict[str, object]:
        """
        End-to-end optimization: prepare inputs, solve, and return a result payload.
        """
        opt_input = self.prepare_optimization_inputs(
            shipments,
            graph=graph,
            alpha=alpha,
            beta=beta,
            cost_per_km=cost_per_km,
            max_route_candidates=max_route_candidates,
            per_trip_overhead_cost=per_trip_overhead_cost,
            per_trip_overhead_emissions_kg=per_trip_overhead_emissions_kg,
        )
        solution = self.run_solver(opt_input)
        return get_optimized_routes_and_reductions(opt_input, solution)

