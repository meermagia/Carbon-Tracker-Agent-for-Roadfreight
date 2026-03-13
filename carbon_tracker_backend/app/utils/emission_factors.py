"""
Emission factor utilities.

Provides emission factors and vehicle capacities by vehicle type for the formula:

    CO2 = Distance × EmissionFactor(vehicle_type) × LoadAdjustment
    LoadAdjustment = shipment_weight / vehicle_capacity

Values are placeholders and should be aligned with accepted standards
(e.g., GHG Protocol, DEFRA, Smart Freight).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleEmissionProfile:
    """Emission profile for a vehicle type."""

    emission_factor_kg_co2e_per_km: float
    """Base emission factor in kg CO2e per km (full-load equivalent)."""

    capacity_tons: float
    """Vehicle capacity in tons for load adjustment."""


# Emission factor (kg CO2e/km) and capacity (tons) per vehicle type
_VEHICLE_PROFILES: dict[str, VehicleEmissionProfile] = {
    "road": VehicleEmissionProfile(emission_factor_kg_co2e_per_km=0.55, capacity_tons=24.0),
    "truck_euro_6": VehicleEmissionProfile(emission_factor_kg_co2e_per_km=0.48, capacity_tons=24.0),
    "truck_euro_5": VehicleEmissionProfile(emission_factor_kg_co2e_per_km=0.60, capacity_tons=24.0),
    "rigid_18t": VehicleEmissionProfile(emission_factor_kg_co2e_per_km=0.35, capacity_tons=18.0),
    "articulated_40t": VehicleEmissionProfile(emission_factor_kg_co2e_per_km=0.65, capacity_tons=40.0),
}


def get_vehicle_profile(vehicle_type: str) -> VehicleEmissionProfile:
    """
    Return the emission profile for a vehicle type.

    Falls back to the default "road" profile if the type is unknown.
    """
    normalized = (vehicle_type or "road").lower().strip()
    return _VEHICLE_PROFILES.get(normalized, _VEHICLE_PROFILES["road"])


def get_emission_factor_for_mode(mode: str) -> float:
    """
    Return the emission factor (kg CO2e per km) for a given vehicle type.

    Kept for backward compatibility; prefer get_vehicle_profile() for new code.
    """
    return get_vehicle_profile(mode).emission_factor_kg_co2e_per_km


# Backwards compatibility constant used by older code paths
# (e.g. `from app.utils.emission_factors import EMISSION_FACTOR`).
# Default to the "road" vehicle profile.
EMISSION_FACTOR: float = _VEHICLE_PROFILES["road"].emission_factor_kg_co2e_per_km

