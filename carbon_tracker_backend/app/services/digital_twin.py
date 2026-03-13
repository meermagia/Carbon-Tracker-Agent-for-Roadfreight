"""
Logistics digital twin simulation module.

This module simulates shipment flows through a logistics network (a NetworkX
graph produced by `graph_builder.py`) in order to estimate emissions under
what-if scenarios before changes are applied operationally.

Scenarios supported:
- Route changes
- Shipment consolidation
- Vehicle type changes

Integration points:
- `carbon_engine`: uses the same emission-factor formula via `get_vehicle_profile`
- `graph_builder`: consumes the logistics graph and its edge distance attribute
- `optimization_engine`: optional hook (route overrides can come from optimizer output)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import networkx as nx
import simpy

from app.utils.emission_factors import VehicleEmissionProfile, get_vehicle_profile

# Integrate with `graph_builder` but keep import optional for lightweight usage.
try:  # pragma: no cover
    from app.services.graph_builder import ATTR_DISTANCE as _GB_ATTR_DISTANCE

    ATTR_DISTANCE = _GB_ATTR_DISTANCE
except Exception:  # pragma: no cover
    # `graph_builder` depends on ORM imports; the graph it produces uses this key.
    ATTR_DISTANCE = "distance"


@dataclass(frozen=True, slots=True)
class ShipmentLike:
    """
    Minimal shipment representation for simulation.

    The backend uses the ORM `Shipment` model; this dataclass exists to make the
    digital twin usable without a database session.
    """

    shipment_id: str
    origin: str
    destination: str
    weight_tons: float
    vehicle_type: str = "road"
    distance_km: float | None = None
    baseline_emissions_kg: float | None = None


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """
    Digital twin scenario configuration.

    Args:
        use_optimized_routes: Reserved flag for higher-level orchestration. The
            simulator itself expects explicit `route_overrides` when applying
            optimizer recommendations.
        route_overrides: Optional mapping shipment_id -> path (list of node labels).
        consolidation: If True, consolidates shipments by (path, vehicle_type) into
            trips constrained by vehicle capacity.
        vehicle_type_overrides: Optional mapping shipment_id -> new vehicle type.
        default_vehicle_type: Optional global override if a shipment has no type.
        edge_capacity: SimPy resource capacity per edge (concurrency). If 1, an
            edge can only serve one "trip" at a time (simple bottleneck model).
        speed_kmph: Travel speed used to convert distance to simulated time.
        per_trip_overhead_emissions_kg: Optional overhead emissions added per trip.
        start_time_fn: Optional function returning a start time (hours) per shipment.
            If None, all trips start at time 0.
    """

    use_optimized_routes: bool = False
    route_overrides: Mapping[str, Sequence[str]] | None = None

    consolidation: bool = False

    vehicle_type_overrides: Mapping[str, str] | None = None
    default_vehicle_type: str = "road"

    edge_capacity: int = 5
    speed_kmph: float = 60.0
    per_trip_overhead_emissions_kg: float = 0.0

    start_time_fn: Callable[[ShipmentLike], float] | None = None


@dataclass(slots=True)
class SimulationContext:
    """Holds the SimPy environment and shared state."""

    env: simpy.Environment
    graph: nx.DiGraph
    edge_resources: Mapping[tuple[str, str], simpy.Resource]
    totals: MutableMapping[str, float]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Results of a simulation run."""

    total_emissions_kg: float
    total_distance_km: float
    trips_simulated: int
    shipments_covered: int
    simulated_hours: float


def _as_shipment_like(obj: Any, *, default_vehicle_type: str = "road") -> ShipmentLike:
    """
    Adapt ORM `Shipment` or dict-like objects into `ShipmentLike`.
    """
    if isinstance(obj, ShipmentLike):
        return obj

    get = (obj.get if isinstance(obj, Mapping) else None)
    if get is not None:
        shipment_id = str(get("shipment_id", get("id", "")))
        origin = str(get("origin_location", get("origin", "")))
        destination = str(get("destination_location", get("destination", "")))
        weight_tons = float(get("weight_tons"))
        vehicle_type = str(get("transport_mode", get("vehicle_type", default_vehicle_type)) or default_vehicle_type)
        distance_km = get("distance_km", None)
        baseline_emissions = get("co2e_kg", get("baseline_emissions_kg", None))
        return ShipmentLike(
            shipment_id=shipment_id,
            origin=origin,
            destination=destination,
            weight_tons=weight_tons,
            vehicle_type=vehicle_type,
            distance_km=(float(distance_km) if distance_km is not None else None),
            baseline_emissions_kg=(float(baseline_emissions) if baseline_emissions is not None else None),
        )

    # Attribute-based (ORM-like)
    shipment_id = str(getattr(obj, "shipment_id", getattr(obj, "id", "")))
    origin = str(getattr(obj, "origin_location", getattr(obj, "origin", "")))
    destination = str(getattr(obj, "destination_location", getattr(obj, "destination", "")))
    weight_tons = float(getattr(obj, "weight_tons"))
    vehicle_type = str(getattr(obj, "transport_mode", getattr(obj, "vehicle_type", default_vehicle_type)) or default_vehicle_type)
    distance_km = getattr(obj, "distance_km", None)
    baseline_emissions = getattr(obj, "co2e_kg", getattr(obj, "baseline_emissions_kg", None))
    return ShipmentLike(
        shipment_id=shipment_id,
        origin=origin,
        destination=destination,
        weight_tons=weight_tons,
        vehicle_type=vehicle_type,
        distance_km=(float(distance_km) if distance_km is not None else None),
        baseline_emissions_kg=(float(baseline_emissions) if baseline_emissions is not None else None),
    )


def _estimate_emissions_kg(distance_km: float, weight_tons: float, profile: VehicleEmissionProfile) -> float:
    """
    Estimate emissions using the same formula as `carbon_engine`:

        CO2e_kg = distance_km * emission_factor_kg_per_km * (weight_tons / capacity_tons)
    """
    load_adjustment = float(weight_tons) / max(float(profile.capacity_tons), 1e-9)
    return float(distance_km) * float(profile.emission_factor_kg_co2e_per_km) * load_adjustment


def _path_distance_km(graph: nx.DiGraph, path: Sequence[str]) -> float:
    total = 0.0
    for u, v in zip(path, path[1:], strict=False):
        if graph.has_edge(u, v):
            total += float(graph.edges[u, v].get(ATTR_DISTANCE, 0.0))
        else:
            raise ValueError(f"Graph missing edge for path segment: {u} -> {v}")
    return total


def initialize_simulation(
    graph: nx.DiGraph,
    *,
    edge_capacity: int = 5,
) -> SimulationContext:
    """
    Initialize a SimPy environment and per-edge resources for the logistics graph.
    """
    env = simpy.Environment()
    cap = int(edge_capacity)
    if cap <= 0:
        raise ValueError("edge_capacity must be >= 1")

    edge_resources: dict[tuple[str, str], simpy.Resource] = {
        (u, v): simpy.Resource(env, capacity=cap) for u, v in graph.edges()
    }
    totals: MutableMapping[str, float] = {"emissions_kg": 0.0, "distance_km": 0.0, "trips": 0.0, "shipments": 0.0}
    return SimulationContext(env=env, graph=graph, edge_resources=edge_resources, totals=totals)


def simulate_shipment_flow(
    ctx: SimulationContext,
    shipment: ShipmentLike,
    *,
    path: Sequence[str],
    vehicle_type: str,
    speed_kmph: float,
    per_trip_overhead_emissions_kg: float = 0.0,
) -> simpy.events.Event:
    """
    SimPy process that simulates a single shipment/trip moving along a path.

    This function yields travel time based on distance and a constant speed, and
    accumulates emissions using the carbon-engine compatible formula.
    """

    def _proc() -> Iterable[simpy.events.Event]:
        profile = get_vehicle_profile(vehicle_type)
        trip_distance = _path_distance_km(ctx.graph, path)

        # Emissions for the whole trip (attributed once per trip).
        trip_emissions = _estimate_emissions_kg(trip_distance, shipment.weight_tons, profile) + float(per_trip_overhead_emissions_kg)

        # Traverse edges; each edge can be capacity constrained.
        for u, v in zip(path, path[1:], strict=False):
            if (u, v) not in ctx.edge_resources:
                raise ValueError(f"No resource for edge {u}->{v}. Did you initialize with the same graph?")
            dist = float(ctx.graph.edges[u, v].get(ATTR_DISTANCE, 0.0))
            travel_hours = dist / max(float(speed_kmph), 1e-9)

            with ctx.edge_resources[(u, v)].request() as req:
                yield req
                yield ctx.env.timeout(travel_hours)

        ctx.totals["emissions_kg"] += float(trip_emissions)
        ctx.totals["distance_km"] += float(trip_distance)
        ctx.totals["trips"] += 1.0
        ctx.totals["shipments"] += 1.0

    return ctx.env.process(_proc())


@dataclass(frozen=True, slots=True)
class _PlannedTrip:
    shipment_id: str
    path: Sequence[str]
    weight_tons: float
    vehicle_type: str
    start_time_hours: float


def _resolve_path_for_shipment(
    graph: nx.DiGraph,
    shipment: ShipmentLike,
    *,
    scenario: ScenarioConfig,
) -> list[str]:
    # Explicit override wins
    if scenario.route_overrides and shipment.shipment_id in scenario.route_overrides:
        p = list(scenario.route_overrides[shipment.shipment_id])
        if len(p) < 2:
            raise ValueError(f"Route override for {shipment.shipment_id} must have at least 2 nodes")
        return p

    # If the direct edge exists, use it.
    if graph.has_edge(shipment.origin, shipment.destination):
        return [shipment.origin, shipment.destination]

    # Otherwise, fall back to shortest path by distance
    try:
        p = nx.shortest_path(graph, shipment.origin, shipment.destination, weight=ATTR_DISTANCE)
        return list(p)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        raise ValueError(f"No path found in graph for {shipment.origin} -> {shipment.destination}") from e


def _plan_trips(
    graph: nx.DiGraph,
    shipments: Sequence[ShipmentLike],
    *,
    scenario: ScenarioConfig,
) -> list[_PlannedTrip]:
    # Apply vehicle type overrides early (affects consolidation capacity/emissions).
    def vehicle_for(s: ShipmentLike) -> str:
        if scenario.vehicle_type_overrides and s.shipment_id in scenario.vehicle_type_overrides:
            return str(scenario.vehicle_type_overrides[s.shipment_id])
        return str(s.vehicle_type or scenario.default_vehicle_type or "road")

    start_fn = scenario.start_time_fn or (lambda _s: 0.0)

    planned = [
        _PlannedTrip(
            shipment_id=s.shipment_id,
            path=_resolve_path_for_shipment(graph, s, scenario=scenario),
            weight_tons=float(s.weight_tons),
            vehicle_type=vehicle_for(s),
            start_time_hours=float(start_fn(s)),
        )
        for s in shipments
    ]

    if not scenario.consolidation:
        return planned

    # Consolidate by (path, vehicle_type, start_time bucket)
    grouped: dict[tuple[tuple[str, ...], str, float], list[_PlannedTrip]] = {}
    for t in planned:
        key = (tuple(t.path), t.vehicle_type, float(t.start_time_hours))
        grouped.setdefault(key, []).append(t)

    consolidated: list[_PlannedTrip] = []
    for (path_key, vehicle_type, start_time), trips in grouped.items():
        profile = get_vehicle_profile(vehicle_type)
        cap = max(float(profile.capacity_tons), 1e-9)

        total_weight = sum(float(t.weight_tons) for t in trips)
        num_trips = int(ceil(total_weight / cap)) if total_weight > 0 else 1

        remaining = float(total_weight)
        for k in range(num_trips):
            w = min(cap, remaining) if remaining > 0 else 0.0
            remaining -= w
            consolidated.append(
                _PlannedTrip(
                    shipment_id=f"consolidated:{trips[0].shipment_id}:{k+1}/{num_trips}",
                    path=list(path_key),
                    weight_tons=float(w),
                    vehicle_type=vehicle_type,
                    start_time_hours=float(start_time),
                )
            )

    return consolidated


def _run_simulation(
    graph: nx.DiGraph,
    shipments: Sequence[ShipmentLike],
    *,
    scenario: ScenarioConfig,
) -> SimulationResult:
    ctx = initialize_simulation(graph, edge_capacity=scenario.edge_capacity)
    planned = _plan_trips(graph, shipments, scenario=scenario)

    for trip in planned:
        # Delay start if needed
        if trip.start_time_hours > 0:
            ctx.env.process(ctx.env.timeout(float(trip.start_time_hours)))

        s = ShipmentLike(
            shipment_id=trip.shipment_id,
            origin=trip.path[0],
            destination=trip.path[-1],
            weight_tons=float(trip.weight_tons),
            vehicle_type=trip.vehicle_type,
        )
        simulate_shipment_flow(
            ctx,
            s,
            path=trip.path,
            vehicle_type=trip.vehicle_type,
            speed_kmph=scenario.speed_kmph,
            per_trip_overhead_emissions_kg=scenario.per_trip_overhead_emissions_kg,
        )

    ctx.env.run()

    return SimulationResult(
        total_emissions_kg=float(ctx.totals["emissions_kg"]),
        total_distance_km=float(ctx.totals["distance_km"]),
        trips_simulated=int(round(float(ctx.totals["trips"]))),
        shipments_covered=int(round(float(ctx.totals["shipments"]))),
        simulated_hours=float(ctx.env.now),
    )


def compare_emission_scenarios(
    graph: nx.DiGraph,
    shipments: Sequence[Any],
    *,
    scenario: ScenarioConfig,
) -> dict[str, Any]:
    """
    Compare baseline vs scenario emissions using the digital twin simulation.

    Baseline interpretation:
    - Uses the current graph and shipment vehicle types
    - Uses direct edge if available, else shortest path by distance
    - If a shipment already has `co2e_kg` (from `CarbonEngine`), it is used for
      baseline totals when available; otherwise baseline is computed with the
      same formula as the carbon engine.
    """
    base_shipments = [_as_shipment_like(s, default_vehicle_type=scenario.default_vehicle_type) for s in shipments]

    # Baseline totals (non-simulated "current configuration")
    baseline_total = 0.0
    for s in base_shipments:
        if s.baseline_emissions_kg is not None:
            baseline_total += float(s.baseline_emissions_kg)
            continue

        # Estimate distance using current graph if possible
        try:
            p = _resolve_path_for_shipment(graph, s, scenario=ScenarioConfig())
            dist = _path_distance_km(graph, p)
        except Exception:
            dist = float(s.distance_km or 0.0)

        profile = get_vehicle_profile(s.vehicle_type or scenario.default_vehicle_type)
        baseline_total += _estimate_emissions_kg(dist, s.weight_tons, profile)

    scenario_result = _run_simulation(graph, base_shipments, scenario=scenario)

    reduction = float(baseline_total) - float(scenario_result.total_emissions_kg)
    reduction_pct = (reduction / float(baseline_total) * 100.0) if baseline_total > 0 else 0.0

    return {
        "baseline": {"total_emissions_kg": float(baseline_total)},
        "scenario": {
            "total_emissions_kg": float(scenario_result.total_emissions_kg),
            "total_distance_km": float(scenario_result.total_distance_km),
            "trips_simulated": int(scenario_result.trips_simulated),
            "shipments_covered": int(scenario_result.shipments_covered),
            "simulated_hours": float(scenario_result.simulated_hours),
        },
        "delta": {
            "emission_reduction_kg": float(reduction),
            "emission_reduction_pct": float(reduction_pct),
        },
    }

