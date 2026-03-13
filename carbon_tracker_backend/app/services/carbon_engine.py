"""
Carbon calculation engine.

Computes shipment-level CO2e using:

    CO2 = Distance × EmissionFactor(vehicle_type) × LoadAdjustment
    LoadAdjustment = shipment_weight / vehicle_capacity

Also aggregates emissions per logistics lane and persists results.
"""

from collections import defaultdict
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.shipment_model import Shipment
from app.models.logistics_lane_model import LogisticsLane
from app.utils.emission_factors import get_vehicle_profile


class CarbonEngine:
    """Engine for computing and aggregating shipment carbon emissions."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def calculate_for_shipment(self, shipment: Shipment) -> float:
        """
        Compute CO2e for a single shipment using:

            CO2 = Distance × EmissionFactor(vehicle_type) × LoadAdjustment
            LoadAdjustment = shipment_weight / vehicle_capacity

        Updates the shipment's co2e_kg and persists to the database.
        """
        profile = get_vehicle_profile(shipment.transport_mode)
        load_adjustment = shipment.weight_tons / profile.capacity_tons

        co2e_kg = (
            shipment.distance_km
            * profile.emission_factor_kg_co2e_per_km
            * load_adjustment
        )

        shipment.co2e_kg = co2e_kg
        self.db.add(shipment)
        self.db.flush()
        return co2e_kg

    def compute_and_store_shipments(
        self, shipments: Iterable[Shipment]
    ) -> list[float]:
        """
        Compute emissions for multiple shipments and store results.

        Returns the list of computed CO2e values (kg) in order.
        """
        results: list[float] = []
        for shipment in shipments:
            co2e = self.calculate_for_shipment(shipment)
            results.append(co2e)
        return results

    def aggregate_emissions_by_lane(
        self, shipments: Iterable[Shipment] | None = None
    ) -> dict[tuple[str, str], dict[str, float | int]]:
        """
        Aggregate CO2e and shipment count per logistics lane (origin, destination).

        If shipments is provided, aggregates from that iterable.
        Otherwise, aggregates from all shipments in the database with co2e_kg set.
        """
        lane_totals: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
            lambda: {"total_co2e_kg": 0.0, "shipment_count": 0}
        )

        if shipments is not None:
            for s in shipments:
                if s.co2e_kg is None:
                    continue
                key = (s.origin_location, s.destination_location)
                lane_totals[key]["total_co2e_kg"] += s.co2e_kg
                lane_totals[key]["shipment_count"] += 1
        else:
            stmt = select(Shipment).where(Shipment.co2e_kg.isnot(None))
            for shipment in self.db.scalars(stmt):
                key = (shipment.origin_location, shipment.destination_location)
                lane_totals[key]["total_co2e_kg"] += shipment.co2e_kg or 0.0
                lane_totals[key]["shipment_count"] += 1

        return dict(lane_totals)

    def store_lane_aggregations(
        self, lane_totals: dict[tuple[str, str], dict[str, float | int]] | None = None
    ) -> int:
        """
        Persist aggregated lane emissions to the logistics_lanes table.

        If lane_totals is not provided, aggregates from the database first.
        Uses upsert logic: updates existing lanes or inserts new ones.
        Returns the number of lanes written.
        """
        if lane_totals is None:
            lane_totals = self.aggregate_emissions_by_lane()

        count = 0
        for (origin, dest), data in lane_totals.items():
            existing = self.db.execute(
                select(LogisticsLane).where(
                    LogisticsLane.origin_location == origin,
                    LogisticsLane.destination_location == dest,
                )
            ).scalars().first()

            total_co2e = float(data["total_co2e_kg"])
            shipment_count = int(data["shipment_count"])

            if existing:
                existing.total_co2e_kg = total_co2e
                existing.shipment_count = shipment_count
                self.db.add(existing)
            else:
                lane = LogisticsLane(
                    origin_location=origin,
                    destination_location=dest,
                    total_co2e_kg=total_co2e,
                    shipment_count=shipment_count,
                )
                self.db.add(lane)
            count += 1

        self.db.flush()
        return count
    
    def compute_aggregate_and_store(self, shipments):
        """
        Compute emissions for multiple shipments
        using the correct vehicle profile logic.
        """

        print("\n========== CARBON ENGINE DEBUG ==========")

        results = []

        for shipment in shipments:
            print(
                f"\nShipment {shipment.shipment_id} | "
                f"Distance={shipment.distance_km} | "
                f"Weight={shipment.weight_tons} | "
                f"Mode={shipment.transport_mode}"
            )

            co2e = self.calculate_for_shipment(shipment)

            print(f"Computed CO2e = {co2e}")

            results.append(co2e)

        print("==========================================\n")

        self.db.flush()
        return results