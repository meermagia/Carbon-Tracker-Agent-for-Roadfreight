## Logistics Carbon Tracker – Backend & Dashboard

An AI‑powered **logistics carbon tracking and optimization** platform for road freight.  
The system ingests shipment data, calculates CO₂e emissions, detects high‑emission lanes, simulates “what‑if” scenarios with a digital twin, and exposes everything via:

- **FastAPI backend** (`carbon_tracker_backend/app`) with ML and optimization services
- **Streamlit dashboard** (`carbon_tracker_backend/dashboard/app.py`) for interactive analytics and simulations

---

## Features

- **Shipment ingestion & storage**
  - Ingest shipments with origin, destination, distance, weight, and transport mode
  - Persist data in **PostgreSQL** via **SQLAlchemy**

- **Carbon accounting**
  - Compute shipment‑level and lane‑level emissions (`co2e_kg`)
  - Aggregate emissions per lane for analytics

- **High‑emission route detection (GNN)**
  - Build a logistics network graph from shipments
  - Use a **Graph Neural Network (GNN)** (GraphSAGE via `torch-geometric`) to flag anomalous / inefficient lanes

- **Time‑series emission forecasting (Transformer)**
  - Per‑lane emission history converted into time‑series
  - Lightweight **Transformer** model predicts future emissions per lane

- **Route optimization (OR‑Tools)**
  - Build a network graph and optimize routing using **Google OR‑Tools**
  - Trade‑off between cost and emissions via tunable weights (`alpha`, `beta`)

- **Digital twin simulation (SimPy)**
  - Simulate logistics operations on a network graph using **SimPy**
  - Compare baseline vs scenario emissions (route changes, consolidation, vehicle type changes)

- **Interactive Streamlit dashboard**
  - **Digital Twin Simulation** UI wired to `POST /simulate_scenario`
  - **Network Map** of Indian cities using `pydeck`
  - **Emissions Analytics**, **Carbon Recommendations**, **Route Optimization**, **GNN Insights**, **Emission Forecast**, and **CO₂ Savings Simulator**

---

## Project Structure

At a high level:

```text
carbon_tracker_backend/
  app/
    api/
      routes.py          # FastAPI API routes
    config.py            # Settings / environment config
    database.py          # SQLAlchemy engine & session
    main.py              # FastAPI application entrypoint
    models/
      shipment_model.py  # Shipment ORM model
      logistics_lane_model.py
    services/
      carbon_engine.py       # Emissions computation
      graph_builder.py       # NetworkX graph construction
      optimization_engine.py # OR-Tools optimization
      digital_twin.py        # SimPy-based simulation
      carbon_heatmap.py
      data_ingestion.py
    ml/
      gnn_model.py           # GNN for high emission lanes
      transformer_model.py   # Time-series emission forecasting
      anomaly_detection.py
    utils/
      emission_factors.py    # Emission factor utilities

  dashboard/
    app.py               # Streamlit dashboard entrypoint

requirements.txt         # Python dependencies for backend & ML
```

---

## Technology Stack

- **Language**: Python 3.10+ (recommended)
- **Web framework**: FastAPI
- **ASGI server**: Uvicorn
- **Database**: PostgreSQL + SQLAlchemy + Alembic (for migrations, if used)
- **ML / Optimization**:
  - PyTorch
  - `torch-geometric` (GNN)
  - `simpy` (discrete‑event simulation)
  - `ortools` (route optimization)
  - `pandas`, `numpy`, `networkx`
- **Dashboard**: Streamlit + Plotly + PyDeck

---

## Prerequisites

- **Python** 3.10 or newer
- **PostgreSQL** (local or remote instance)
- **Git** (for version control / cloning)

Optional but recommended:

- `virtualenv` or `conda` for isolated Python environments

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>.git
cd "LoRri Zip/Carbon Tracker Agent for Roadfreight"
```

Adjust the path to match your local folder name if different.

### 2. Create and activate a virtual environment

On **Windows (PowerShell)**:

```bash
python -m venv venv
venv\Scripts\activate
```

On **macOS / Linux**:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

From the project root:

```bash
cd carbon_tracker_backend
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: Installing `torch` / `torch-geometric` may require platform‑specific wheels.  
> If `pip install` fails, consult the official installation guides for your OS / CUDA setup.

---

## Configuration

Backend configuration lives in `app/config.py` using `pydantic-settings`.  
By default it loads environment variables from `carbon_tracker_backend/.env`.

### Environment variables

Key settings (with defaults from `Settings` in `app/config.py`):

- **`APP_NAME`** (optional) – defaults to `"Carbon Tracker Backend"`
- **`ENVIRONMENT`** (optional) – e.g. `"development"`, `"production"`
- **`DEBUG`** (optional) – `True` / `False`
- **`DB_HOST`** – default: `"localhost"`
- **`DB_PORT`** – default: `5432`
- **`DB_NAME`** – default: `"carbon_tracker"`
- **`DB_USER`** – default: `"postgres"`
- **`DB_PASSWORD`** – default: `"user"`

The database URL is built as:

```text
postgresql+psycopg2://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME
```

### Example `.env`

Create `carbon_tracker_backend/.env`:

```env
APP_NAME="Carbon Tracker Backend"
ENVIRONMENT="development"
DEBUG=true

DB_HOST=localhost
DB_PORT=5432
DB_NAME=carbon_tracker
DB_USER=postgres
DB_PASSWORD=your_password_here
```

Make sure the corresponding PostgreSQL database and user exist and have access.

---

## Running the FastAPI Backend

From the `carbon_tracker_backend` directory (with virtualenv activated and `.env` configured):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup:

- Tables are created (via `Base.metadata.create_all`)
- A test `SELECT 1` is executed to confirm DB connectivity

### API Docs

Once running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

All routes are prefixed with `/api/v1`.

---

## Running the Streamlit Dashboard

The dashboard in `carbon_tracker_backend/dashboard/app.py` talks to the backend via HTTP.

1. Ensure the **backend** is running (see previous section).
2. In another terminal, from `carbon_tracker_backend`:

```bash
streamlit run dashboard/app.py
```

By default, the dashboard:

- Uses `st.secrets["API_BASE_URL"]` if provided, otherwise defaults to `http://localhost:8000`
- Calls endpoints like `/emissions`, `/simulate_scenario`, `/high_emission_routes`, `/predict_emissions`, `/optimize_routes`

You can override the API URL from the **sidebar** in the UI.

### Optional: `secrets.toml` for Streamlit

Create `.streamlit/secrets.toml` inside `carbon_tracker_backend/dashboard`:

```toml
API_BASE_URL = "http://localhost:8000/api/v1"
```

> Note: The dashboard usually appends relative paths (e.g. `/emissions`) to this base URL.  
> Ensure the prefix (`/api/v1` vs none) matches how you’re deploying the backend.

---

## Key API Endpoints (Summary)

All routes below are under the prefix `/api/v1`.

### System

- **`GET /health`**
  - Simple health check; verifies that the API is running and the database is reachable.

### Shipment ingestion & emissions

- **`POST /ingest_shipment`**
  - Body: `ShipmentCreateRequest`
    - `shipment_id: str`
    - `origin_location: str`
    - `destination_location: str`
    - `distance_km: float`
    - `weight_tons: float`
    - `transport_mode: str` (default: `"road"`)
  - Response: `IngestShipmentResponse`
  - Behavior:
    - Stores a new shipment (409 if `shipment_id` already exists)
    - Computes and stores `co2e_kg` via `CarbonEngine`

- **`GET /emissions`**
  - Query params:
    - `recompute_missing: bool = true`
  - Response: `EmissionsResponse` with:
    - List of `ShipmentResponse`
    - `total_emissions_kg`

### High‑emission route detection (GNN)

- **`GET /high_emission_routes`**
  - Query params:
    - `epochs: int = 50`
    - `z_threshold: float = 2.0`
  - Behavior:
    - Builds a logistics graph from shipments
    - Computes lane‐level carbon intensity
    - Trains a small GNN (GraphSAGE) and flags anomalous / inefficient lanes
  - Response: `HighEmissionRoutesResponse`
    - `anomalies: List[Dict]`
    - `num_nodes`, `num_edges`

### Emission forecasting (Transformer)

- **`GET /predict_emissions`**
  - Query params:
    - `input_length: int = 4`
    - `horizon: int = 3`
    - `epochs: int = 50`
  - Behavior:
    - Builds lane‑wise time‐series from historical shipments
    - Trains a Transformer on emissions sequences and forecasts the next `horizon` steps
    - Returns empty predictions when there is not enough history (instead of failing)
  - Response: `PredictEmissionsResponse`
    - `horizon: int`
    - `predictions_by_lane: Dict[str, List[float]]`

### Route optimization (OR‑Tools)

- **`POST /optimize_routes`**
  - Body: `OptimizeRoutesRequest`
    - `shipment_ids: Optional[List[str]]`
    - `alpha: float` (emissions weight)
    - `beta: float` (cost weight)
    - `cost_per_km: float`
    - `max_route_candidates: int`
    - `per_trip_overhead_cost: float`
    - `per_trip_overhead_emissions_kg: float`
  - Behavior:
    - Fetches shipments (filtered by IDs if provided)
    - Recomputes emissions via `CarbonEngine`
    - Builds graph via `GraphBuilder` and runs optimization via `OptimizationEngine`
  - Response: `OptimizeRoutesResponse` with a `result` dict that includes (among others):
    - `baseline_emissions`
    - `totals.baseline_emissions_kg`
    - `totals.optimized_emissions_kg`
    - `totals.emission_reduction_pct`

### Digital twin simulation

- **`POST /simulate_scenario`**
  - Body: `SimulateScenarioRequest`
    - `shipment_ids: Optional[List[str]]`
    - `route_changes: RouteChangesPayload`
      - `overrides_by_shipment_id: Optional[Dict[str, List[str]]]`
      - `optimize_with_engine: bool`
      - plus optimization parameters (alpha, beta, cost_per_km, etc.) when `optimize_with_engine = true`
    - `consolidation: ConsolidationPayload`
      - `enabled: bool`
      - `per_trip_overhead_emissions_kg: float`
    - `vehicle_type_changes: VehicleTypeChangesPayload`
      - `overrides_by_shipment_id: Optional[Dict[str, str]]`
    - Simulation controls:
      - `edge_capacity: int`
      - `speed_kmph: float`
  - Behavior:
    - Ensures `co2e_kg` is available for all shipments (baseline)
    - Builds graph via `GraphBuilder`
    - Optionally runs `OptimizationEngine` and applies optimized routes as overrides
    - Runs scenario via `compare_emission_scenarios` from `digital_twin`
  - Response: `SimulateScenarioResponse`
    - `baseline_emissions_kg`
    - `simulated_emissions_kg`
    - `emission_reduction_kg`
    - `emission_reduction_pct`
    - `simulation_metrics` (raw scenario metrics)
    - `recommended_changes` with:
      - Applied route changes
      - Applied vehicle changes
      - Consolidation flag
      - Optimizer result (if used)

---

## Using the Streamlit Dashboard

Once the backend and dashboard are running:

- Open the Streamlit URL (typically `http://localhost:8501`)
- In the **sidebar**:
  - Set **API Base URL** to the backend (e.g. `http://localhost:8000/api/v1` or `http://localhost:8000`)
  - Use the navigation buttons:
    - **🧪 Digital Twin Simulation** – per‑route scenario analysis
    - **🌍 Network Map** – visualize flows between major Indian cities
    - **📊 Emissions Analytics** – metrics, top emitting routes, CSV export
    - **🌱 Carbon Recommendations** – suggested shifts (e.g. diesel → electric)
    - **🚀 Route Optimization** – call `POST /optimize_routes` from the UI
    - **🧠 GNN Insights** – high‑emission routes from the GNN
    - **📈 Emission Forecast** – aggregated forecast from the Transformer
    - **⚡ CO₂ Savings Simulator** – simple what‑if calculator

Most pages pull from the same shipment/emissions API, so make sure you have data ingested first.

---

## Development Notes

- **Database migrations**: If you introduce schema changes, consider adding Alembic migrations under `carbon_tracker_backend`.
- **Model training in production**: The GNN and Transformer models are trained on the fly in this reference implementation. For production, you would typically:
  - Pre‑train and version models offline
  - Load them at startup instead of re‑training per request
  - Add caching / background jobs for heavy computations
- **Logging & monitoring**: For local development, `print` statements are used in some services (e.g. optimization). In a real deployment, replace these with structured logging.

---

## Testing the API Quickly

With backend running, you can use `curl` or any REST client.

### Example: ingest a shipment

```bash
curl -X POST "http://localhost:8000/api/v1/ingest_shipment" ^
  -H "Content-Type: application/json" ^
  -d "{
    \"shipment_id\": \"SHP-001\",
    \"origin_location\": \"Mumbai\",
    \"destination_location\": \"Delhi\",
    \"distance_km\": 1400,
    \"weight_tons\": 10,
    \"transport_mode\": \"road\"
  }"
```

### Example: fetch emissions

```bash
curl "http://localhost:8000/api/v1/emissions"
```

---

