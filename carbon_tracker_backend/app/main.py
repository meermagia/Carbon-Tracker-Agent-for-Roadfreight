from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text

from app.api.routes import router as api_router
from app.config import get_settings
from app.database import engine, Base

# Force model imports so SQLAlchemy registers them
import app.models.shipment_model
import app.models.logistics_lane_model  


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print("✅ Database connected and tables created")

    except Exception as e:
        print("❌ Database startup error:", e)

    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")