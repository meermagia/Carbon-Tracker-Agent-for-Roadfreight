from datetime import datetime

from sqlalchemy import String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Shipment(Base):
    """
    Shipment entity representing a road freight movement.

    This will be the primary unit for:
    - data ingestion
    - carbon emission calculation
    - graph construction
    - optimization and analytics
    """

    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    shipment_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    origin_location: Mapped[str] = mapped_column(String(128), index=True)
    destination_location: Mapped[str] = mapped_column(String(128), index=True)

    distance_km: Mapped[float] = mapped_column(Float)
    weight_tons: Mapped[float] = mapped_column(Float)
    transport_mode: Mapped[str] = mapped_column(String(32), default="road")

    co2e_kg: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

