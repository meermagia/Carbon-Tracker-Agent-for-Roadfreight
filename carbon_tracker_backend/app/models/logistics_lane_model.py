"""
Logistics lane model for aggregated emissions per origin–destination pair.
"""

from datetime import datetime

from sqlalchemy import String, Float, DateTime, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class LogisticsLane(Base):
    """
    Aggregated emissions for a logistics lane (origin → destination).

    Used for reporting and analytics on carbon emissions per route.
    """

    __tablename__ = "logistics_lanes"
    __table_args__ = (
        UniqueConstraint("origin_location", "destination_location", name="uq_lane_origin_dest"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    origin_location: Mapped[str] = mapped_column(String(128), index=True)
    destination_location: Mapped[str] = mapped_column(String(128), index=True)

    total_co2e_kg: Mapped[float] = mapped_column(Float, default=0.0)
    shipment_count: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
