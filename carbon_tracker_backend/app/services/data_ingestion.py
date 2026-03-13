"""
Data ingestion service.

Responsible for:
- Loading shipment data from CSV/Parquet/JSON or APIs
- Validating and normalizing fields
- Persisting shipments into the database
"""

from typing import Protocol

import pandas as pd
from sqlalchemy.orm import Session

from app.models.shipment_model import Shipment


class ShipmentIngestionLogger(Protocol):
    def info(self, msg: str, *args, **kwargs) -> None: ...

    def warning(self, msg: str, *args, **kwargs) -> None: ...

    def error(self, msg: str, *args, **kwargs) -> None: ...


class DataIngestionService:
    def __init__(self, db: Session, logger: ShipmentIngestionLogger | None = None) -> None:
        self.db = db
        self.logger = logger

    def ingest_dataframe(self, df: pd.DataFrame) -> int:
        """
        Ingest shipment records from a pandas DataFrame.

        This is a placeholder; detailed schema mapping and validation
        will be implemented later.
        """
        if df.empty:
            return 0

        # TODO: implement real ingestion logic
        if self.logger:
            self.logger.info("Received DataFrame with %d rows for ingestion", len(df))

        # No-op for now
        return 0

