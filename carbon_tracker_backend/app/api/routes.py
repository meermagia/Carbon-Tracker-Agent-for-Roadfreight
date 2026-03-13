from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.shipment_model import Shipment
from app.services.carbon_engine import CarbonEngine
from app.services.graph_builder import (
    GraphBuilder,
    compute_lane_carbon_intensity,
)
from app.services.optimization_engine import OptimizationEngine
from app.services.digital_twin import ScenarioConfig, compare_emission_scenarios

router = APIRouter()


class ShipmentCreateRequest(BaseModel):
    """Request model for ingesting a shipment."""

    shipment_id: str = Field(..., min_length=1, max_length=64)
    origin_location: str = Field(..., min_length=1, max_length=128)
    destination_location: str = Field(..., min_length=1, max_length=128)
    distance_km: float = Field(..., gt=0)
    weight_tons: float = Field(..., gt=0)
    transport_mode: str = Field(default="road", min_length=1, max_length=32)


class ShipmentResponse(BaseModel):
    """Response model for a shipment record."""

    shipment_id: str
    origin_location: str
    destination_location: str
    distance_km: float
    weight_tons: float
    transport_mode: str
    co2e_kg: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class IngestShipmentResponse(BaseModel):
    """Response model for shipment ingestion."""

    status: str
    shipment: ShipmentResponse


class EmissionsResponse(BaseModel):
    """Response model for emission listing."""

    shipments: List[ShipmentResponse]
    total_emissions_kg: float


class HighEmissionRoutesResponse(BaseModel):
    """Response model for high emission / inefficient route detection."""

    anomalies: List[Dict[str, Any]]
    num_nodes: int
    num_edges: int


class PredictEmissionsResponse(BaseModel):
    """Response model for emission forecasting."""

    horizon: int
    predictions_by_lane: Dict[str, List[float]]


class OptimizeRoutesRequest(BaseModel):
    """Request model for route optimization."""

    shipment_ids: Optional[List[str]] = None
    alpha: float = Field(default=1.0, ge=0)
    beta: float = Field(default=1.0, ge=0)
    cost_per_km: float = Field(default=1.0, ge=0)
    max_route_candidates: int = Field(default=2, ge=1, le=5)
    per_trip_overhead_cost: float = Field(default=0.0, ge=0)
    per_trip_overhead_emissions_kg: float = Field(default=0.0, ge=0)


class OptimizeRoutesResponse(BaseModel):
    """Response model for route optimization."""

    result: Dict[str, Any]


class RouteChangesPayload(BaseModel):
    """
    Scenario route changes.

    - If `overrides_by_shipment_id` is provided, those paths are applied directly.
    - If `optimize_with_engine` is true, the optimization engine is used to recommend
      routes and the recommendations are applied as overrides for simulation.
    """

    overrides_by_shipment_id: Optional[Dict[str, List[str]]] = None
    optimize_with_engine: bool = False

    # Optimization parameters (only used when optimize_with_engine=True)
    alpha: float = Field(default=1.0, ge=0)
    beta: float = Field(default=1.0, ge=0)
    cost_per_km: float = Field(default=1.0, ge=0)
    max_route_candidates: int = Field(default=2, ge=1, le=5)
    per_trip_overhead_cost: float = Field(default=0.0, ge=0)
    per_trip_overhead_emissions_kg: float = Field(default=0.0, ge=0)

    @field_validator("overrides_by_shipment_id")
    @classmethod
    def _validate_route_overrides(
        cls, v: Optional[Dict[str, List[str]]]
    ) -> Optional[Dict[str, List[str]]]:
        if v is None:
            return None
        for sid, path in v.items():
            if not sid or not str(sid).strip():
                raise ValueError("Route override shipment_id must be a non-empty string")
            if not isinstance(path, list) or len(path) < 2:
                raise ValueError(f"Route override for {sid} must be a list of at least 2 node labels")
            for node in path:
                if not str(node).strip():
                    raise ValueError(f"Route override for {sid} contains an empty node label")
        return v


class ConsolidationPayload(BaseModel):
    """Scenario shipment consolidation parameters."""

    enabled: bool = False
    per_trip_overhead_emissions_kg: float = Field(default=0.0, ge=0)


class VehicleTypeChangesPayload(BaseModel):
    """Scenario vehicle type changes by shipment id."""

    overrides_by_shipment_id: Optional[Dict[str, str]] = None

    @field_validator("overrides_by_shipment_id")
    @classmethod
    def _validate_vehicle_overrides(
        cls, v: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        if v is None:
            return None
        for sid, vt in v.items():
            if not sid or not str(sid).strip():
                raise ValueError("Vehicle override shipment_id must be a non-empty string")
            if not vt or not str(vt).strip():
                raise ValueError(f"Vehicle override for {sid} must be a non-empty vehicle type")
        return v


class SimulateScenarioRequest(BaseModel):
    """Request model for digital twin scenario simulation."""

    shipment_ids: Optional[List[str]] = None
    route_changes: RouteChangesPayload = Field(default_factory=RouteChangesPayload)
    consolidation: ConsolidationPayload = Field(default_factory=ConsolidationPayload)
    vehicle_type_changes: VehicleTypeChangesPayload = Field(default_factory=VehicleTypeChangesPayload)

    # Simulation controls
    edge_capacity: int = Field(default=5, ge=1, le=1000)
    speed_kmph: float = Field(default=60.0, gt=0)


class AppliedRouteChange(BaseModel):
    shipment_id: str
    path: List[str]


class AppliedVehicleChange(BaseModel):
    shipment_id: str
    from_vehicle_type: str
    to_vehicle_type: str


class RecommendedChangesSummary(BaseModel):
    """
    Summary of changes applied/recommended for the dashboard.
    """

    route_changes_applied: List[AppliedRouteChange]
    vehicle_type_changes_applied: List[AppliedVehicleChange]
    consolidation_enabled: bool
    optimizer_result: Optional[Dict[str, Any]] = None


class SimulateScenarioResponse(BaseModel):
    """Response model for scenario simulation."""

    baseline_emissions_kg: float
    simulated_emissions_kg: float
    emission_reduction_kg: float
    emission_reduction_pct: float
    simulation_metrics: Dict[str, Any]
    recommended_changes: RecommendedChangesSummary


@router.get("/health", tags=["system"])
def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Basic health check endpoint.

    Verifies that the API is running and the database is reachable.
    """
    db.execute(text("SELECT 1"))
    settings = get_settings()
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@router.post("/ingest_shipment", response_model=IngestShipmentResponse, tags=["ingestion"])
def ingest_shipment(payload: ShipmentCreateRequest, db: Session = Depends(get_db)) -> IngestShipmentResponse:
    """
    Ingest a shipment record and store it in the database.

    Also computes and stores shipment emissions and updates lane aggregation.
    """
    existing = db.execute(select(Shipment).where(Shipment.shipment_id == payload.shipment_id)).scalars().first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Shipment with this shipment_id already exists")

    shipment = Shipment(
        shipment_id=payload.shipment_id,
        origin_location=payload.origin_location,
        destination_location=payload.destination_location,
        distance_km=payload.distance_km,
        weight_tons=payload.weight_tons,
        transport_mode=payload.transport_mode,
    )
    db.add(shipment)
    db.flush()

    carbon_engine = CarbonEngine(db)
    carbon_engine.compute_aggregate_and_store([shipment])

    return IngestShipmentResponse(
        status="stored",
        shipment=ShipmentResponse(
            shipment_id=shipment.shipment_id,
            origin_location=shipment.origin_location,
            destination_location=shipment.destination_location,
            distance_km=float(shipment.distance_km),
            weight_tons=float(shipment.weight_tons),
            transport_mode=shipment.transport_mode,
            co2e_kg=float(shipment.co2e_kg) if shipment.co2e_kg is not None else None,
            created_at=shipment.created_at,
            updated_at=shipment.updated_at,
        ),
    )


@router.get("/emissions", response_model=EmissionsResponse, tags=["emissions"])
def get_emissions(
    db: Session = Depends(get_db),
    recompute_missing: bool = Query(default=True, description="Compute emissions for shipments without co2e_kg"),
) -> EmissionsResponse:
    """
    Return calculated shipment emissions using the carbon engine.
    """
    shipments = list(db.scalars(select(Shipment)).all())
    carbon_engine = CarbonEngine(db)

    if recompute_missing:
        missing = [s for s in shipments if s.co2e_kg is None]
        if missing:
            carbon_engine.compute_aggregate_and_store(missing)

    total = 0.0
    out: List[ShipmentResponse] = []
    for s in shipments:
        if s.co2e_kg is not None:
            total += float(s.co2e_kg)
        out.append(
            ShipmentResponse(
                shipment_id=s.shipment_id,
                origin_location=s.origin_location,
                destination_location=s.destination_location,
                distance_km=float(s.distance_km),
                weight_tons=float(s.weight_tons),
                transport_mode=s.transport_mode,
                co2e_kg=float(s.co2e_kg) if s.co2e_kg is not None else None,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
        )

    return EmissionsResponse(shipments=out, total_emissions_kg=float(total))


@router.get("/high_emission_routes", response_model=HighEmissionRoutesResponse, tags=["routes"])
def high_emission_routes(
    db: Session = Depends(get_db),
    epochs: int = Query(default=50, ge=1, le=500),
    z_threshold: float = Query(default=2.0, ge=0.5, le=10.0),
) -> HighEmissionRoutesResponse:
    """
    Detect inefficient or high-emission logistics lanes using the GNN module.

    This endpoint trains a lightweight GraphSAGE model on the current graph's
    edge carbon intensity as a proxy target, then flags anomalous lanes by score.
    In production, replace this with loading a pre-trained model.
    """
    shipments = list(db.scalars(select(Shipment)).all())
    if not shipments:
        return HighEmissionRoutesResponse(anomalies=[], num_nodes=0, num_edges=0)

    # Ensure emissions are available
    carbon_engine = CarbonEngine(db)
    missing = [s for s in shipments if s.co2e_kg is None]
    if missing:
        carbon_engine.compute_aggregate_and_store(missing)

    builder = GraphBuilder()
    graph = builder.build_from_shipments(shipments)
    compute_lane_carbon_intensity(graph)

    import torch
    from app.ml.gnn_model import (
        GNNConfig,
        RouteGNN,
        detect_inefficient_lanes,
        networkx_to_pyg,
        train_route_gnn,
    )

    data, _ = networkx_to_pyg(graph)
    if data.edge_index.numel() == 0:
        return HighEmissionRoutesResponse(anomalies=[], num_nodes=int(data.num_nodes), num_edges=0)

    model = RouteGNN(in_node_channels=data.x.size(-1), in_edge_channels=data.edge_attr.size(-1), config=GNNConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_route_gnn(model, data, num_epochs=epochs, optimizer=optimizer)

    anomalies = detect_inefficient_lanes(graph, model, z_threshold=z_threshold)
    return HighEmissionRoutesResponse(
        anomalies=anomalies,
        num_nodes=int(data.num_nodes),
        num_edges=int(data.edge_index.size(1)),
    )


@router.get("/predict_emissions", response_model=PredictEmissionsResponse, tags=["forecasting"])
def predict_lane_emissions(
    db: Session = Depends(get_db),
    input_length: int = Query(default=4, ge=2, le=365),
    horizon: int = Query(default=3, ge=1, le=90),
    epochs: int = Query(default=50, ge=1, le=500),
) -> PredictEmissionsResponse:
    """
    Forecast future emissions for each lane based on historical shipment data.

    This endpoint trains a lightweight Transformer model on per-lane emission
    time-series derived from shipments ordered by created_at, then predicts the
    next `horizon` steps for each lane.
    In production, replace this with loading a pre-trained model and a stable
    lane time-bucketing strategy (e.g., daily totals).
    """
    shipments = list(db.scalars(select(Shipment).order_by(Shipment.created_at.asc())).all())
    if not shipments:
        return PredictEmissionsResponse(horizon=horizon, predictions_by_lane={})

    # Ensure emissions are available
    carbon_engine = CarbonEngine(db)
    missing = [s for s in shipments if s.co2e_kg is None]
    if missing:
        carbon_engine.compute_aggregate_and_store(missing)

    # Build lane time-series: lane_id -> emissions list (ordered by created_at)
    lane_series: Dict[str, List[float]] = {}
    for s in shipments:
        lane_id = f"{s.origin_location}|{s.destination_location}"
        lane_series.setdefault(lane_id, []).append(float(s.co2e_kg or 0.0))

    import torch
    from torch.utils.data import DataLoader

    from app.ml.transformer_model import (
        EmissionTransformer,
        TimeSeriesConfig,
        predict_emissions,
        prepare_emission_timeseries_dataset,
        train_emission_transformer,
    )

    config = TimeSeriesConfig(input_length=input_length, forecast_horizon=horizon)
    dataset, _ = prepare_emission_timeseries_dataset(
        lane_series, input_length=config.input_length, forecast_horizon=config.forecast_horizon
    )

    # If there is not enough history to build any training windows,
    # return an empty forecast instead of an HTTP 400 so the dashboard
    # can show a friendly "No forecast data available" message.
    if len(dataset) == 0:
        return PredictEmissionsResponse(horizon=horizon, predictions_by_lane={})

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = EmissionTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_emission_transformer(model, loader, num_epochs=epochs, optimizer=optimizer)

    # Inference: for each lane, take last input_length values
    predictions: Dict[str, List[float]] = {}
    for lane_id, series in lane_series.items():
        if len(series) < input_length:
            continue
        history = torch.tensor(series[-input_length:], dtype=torch.float32).view(1, input_length, 1)
        forecast = predict_emissions(model, history).squeeze(0).squeeze(-1).cpu().tolist()
        predictions[lane_id] = [float(x) for x in forecast]

    return PredictEmissionsResponse(horizon=horizon, predictions_by_lane=predictions)

@router.post("/optimize_routes", response_model=OptimizeRoutesResponse, tags=["optimization"])
def optimize_routes(payload: OptimizeRoutesRequest, db: Session = Depends(get_db)) -> OptimizeRoutesResponse:
    """
    Suggest improved routing strategies using the OR-Tools optimization engine.
    """

    # -----------------------------
    # 1️⃣ Fetch Shipments
    # -----------------------------
    stmt = select(Shipment)

    if payload.shipment_ids:
        stmt = stmt.where(Shipment.shipment_id.in_(payload.shipment_ids))

    shipments = list(db.scalars(stmt).all())

    if not shipments:
        raise HTTPException(status_code=404, detail="No shipments found for optimization")

    print("\n========== SHIPMENTS FETCHED ==========")
    for s in shipments:
        print(
            f"Shipment {s.shipment_id} | "
            f"Distance={s.distance_km} | "
            f"Weight={s.weight_tons} | "
            f"Stored CO2={s.co2e_kg}"
        )
    print("=======================================\n")

    # -----------------------------
    # 2️⃣ FORCE EMISSION RECALCULATION
    # -----------------------------
    carbon_engine = CarbonEngine(db)

    print("========== FORCING EMISSION RECALCULATION ==========")
    carbon_engine.compute_aggregate_and_store(shipments)
    db.commit()

    # Refresh values from DB
    for shipment in shipments:
        db.refresh(shipment)
        print(
            f"Shipment {shipment.shipment_id} | "
            f"New CO2={shipment.co2e_kg}"
        )
    print("=====================================================\n")

    # -----------------------------
    # 3️⃣ Compute Baseline Emissions
    # -----------------------------
    baseline_emissions = sum(s.co2e_kg or 0 for s in shipments)

    print("========== BASELINE DEBUG ==========")
    print(f"Baseline Emissions = {baseline_emissions}")
    print("=====================================\n")

    # -----------------------------
    # 4️⃣ Build Graph
    # -----------------------------
    builder = GraphBuilder()
    graph = builder.build_from_shipments(shipments)

    compute_lane_carbon_intensity(graph)

    # -----------------------------
    # 5️⃣ Run Optimization Engine
    # -----------------------------
    engine = OptimizationEngine()

    result = engine.optimize(
        shipments,
        graph=graph,
        alpha=payload.alpha,
        beta=payload.beta,
        cost_per_km=payload.cost_per_km,
        max_route_candidates=payload.max_route_candidates,
        per_trip_overhead_cost=payload.per_trip_overhead_cost,
        per_trip_overhead_emissions_kg=payload.per_trip_overhead_emissions_kg,
    )

    print("========== OPTIMIZATION RESULT ==========")
    print(result)
    print("==========================================\n")

    # -----------------------------
    # 6️⃣ Ensure Baseline Is Attached
    # -----------------------------
    # (In case your OptimizationEngine does not return it)
    result["baseline_emissions"] = baseline_emissions

    return OptimizeRoutesResponse(result=result)


@router.post("/simulate_scenario", response_model=SimulateScenarioResponse, tags=["simulation"])
def simulate_scenario(payload: SimulateScenarioRequest, db: Session = Depends(get_db)) -> SimulateScenarioResponse:
    """
    Simulate a logistics scenario using the digital twin (SimPy) and compare emissions.

    Integrations:
    - carbon_engine: ensures shipment `co2e_kg` is available for baseline comparison
    - graph_builder: builds the logistics network graph
    - optimization_engine: optional route recommendations (applied as overrides)
    - digital_twin: runs the scenario simulation and returns the comparison metrics
    """
    stmt = select(Shipment)
    if payload.shipment_ids:
        stmt = stmt.where(Shipment.shipment_id.in_(payload.shipment_ids))
    shipments = list(db.scalars(stmt).all())
    if not shipments:
        raise HTTPException(status_code=404, detail="No shipments found for simulation")

    # Ensure emissions are available (baseline uses shipment.co2e_kg when present)
    carbon_engine = CarbonEngine(db)
    missing = [s for s in shipments if s.co2e_kg is None]
    if missing:
        carbon_engine.compute_aggregate_and_store(missing)

    # Build graph for scenario routing.
    builder = GraphBuilder()
    graph = builder.build_from_shipments(shipments)
    compute_lane_carbon_intensity(graph)

    # Route overrides either come directly from payload or via the optimization engine.
    route_overrides: Dict[str, List[str]] = {}
    optimizer_result: Optional[Dict[str, Any]] = None

    if payload.route_changes.overrides_by_shipment_id:
        route_overrides.update(payload.route_changes.overrides_by_shipment_id)

    if payload.route_changes.optimize_with_engine:
        engine = OptimizationEngine()
        optimizer_result = engine.optimize(
            shipments,
            graph=graph,
            alpha=payload.route_changes.alpha,
            beta=payload.route_changes.beta,
            cost_per_km=payload.route_changes.cost_per_km,
            max_route_candidates=payload.route_changes.max_route_candidates,
            per_trip_overhead_cost=payload.route_changes.per_trip_overhead_cost,
            per_trip_overhead_emissions_kg=payload.route_changes.per_trip_overhead_emissions_kg,
        )
        for item in optimizer_result.get("optimized_assignments", []):
            sid = str(item.get("shipment_id"))
            path = item.get("path")
            if sid and isinstance(path, list) and len(path) >= 2:
                route_overrides[sid] = [str(x) for x in path]

    vehicle_overrides = payload.vehicle_type_changes.overrides_by_shipment_id or {}

    # Build scenario config for the digital twin.
    scenario_cfg = ScenarioConfig(
        route_overrides=route_overrides or None,
        consolidation=bool(payload.consolidation.enabled),
        vehicle_type_overrides=vehicle_overrides or None,
        edge_capacity=int(payload.edge_capacity),
        speed_kmph=float(payload.speed_kmph),
        per_trip_overhead_emissions_kg=float(payload.consolidation.per_trip_overhead_emissions_kg),
    )

    comparison = compare_emission_scenarios(graph, shipments, scenario=scenario_cfg)

    baseline = float(comparison["baseline"]["total_emissions_kg"])
    simulated = float(comparison["scenario"]["total_emissions_kg"])
    reduction_kg = float(comparison["delta"]["emission_reduction_kg"])
    reduction_pct = float(comparison["delta"]["emission_reduction_pct"])

    # Summarize changes (dashboard-friendly)
    route_changes_applied = [
        AppliedRouteChange(shipment_id=sid, path=path) for sid, path in sorted(route_overrides.items())
    ]
    vehicle_changes_applied: List[AppliedVehicleChange] = []
    if vehicle_overrides:
        by_id = {s.shipment_id: s for s in shipments}
        for sid, to_type in sorted(vehicle_overrides.items()):
            s = by_id.get(sid)
            if s is None:
                continue
            vehicle_changes_applied.append(
                AppliedVehicleChange(
                    shipment_id=sid,
                    from_vehicle_type=str(s.transport_mode),
                    to_vehicle_type=str(to_type),
                )
            )

    metrics = dict(comparison["scenario"])

    return SimulateScenarioResponse(
        baseline_emissions_kg=baseline,
        simulated_emissions_kg=simulated,
        emission_reduction_kg=reduction_kg,
        emission_reduction_pct=reduction_pct,
        simulation_metrics=metrics,
        recommended_changes=RecommendedChangesSummary(
            route_changes_applied=route_changes_applied,
            vehicle_type_changes_applied=vehicle_changes_applied,
            consolidation_enabled=bool(payload.consolidation.enabled),
            optimizer_result=optimizer_result,
        ),
    )

