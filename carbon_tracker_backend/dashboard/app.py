"""
Streamlit dashboard for the Logistics Carbon Tracker.

Includes a "Digital Twin Simulation" section that calls the FastAPI backend
`POST /simulate_scenario` endpoint to compare baseline vs simulated emissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import plotly.graph_objects as go
import requests
import streamlit as st


@dataclass(frozen=True, slots=True)
class ShipmentRow:
    shipment_id: str
    origin: str
    destination: str
    transport_mode: str
    co2e_kg: float | None


def _api_url_default() -> str:
    return st.secrets.get("API_BASE_URL", "http://localhost:8000")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@st.cache_data(ttl=15)
def fetch_shipments(api_base_url: str) -> List[ShipmentRow]:
    """
    Fetch shipments + emissions from the backend.
    """
    url = f"{api_base_url.rstrip('/')}/emissions"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    shipments = data.get("shipments", []) or []
    out: List[ShipmentRow] = []
    for s in shipments:
        out.append(
            ShipmentRow(
                shipment_id=str(s.get("shipment_id")),
                origin=str(s.get("origin_location")),
                destination=str(s.get("destination_location")),
                transport_mode=str(s.get("transport_mode") or "road"),
                co2e_kg=_safe_float(s.get("co2e_kg")),
            )
        )
    return out


def lane_key(origin: str, destination: str) -> str:
    return f"{origin} → {destination}"


def build_lane_index(shipments: Iterable[ShipmentRow]) -> Dict[str, List[ShipmentRow]]:
    lanes: Dict[str, List[ShipmentRow]] = {}
    for s in shipments:
        lanes.setdefault(lane_key(s.origin, s.destination), []).append(s)
    for k in list(lanes.keys()):
        lanes[k] = sorted(lanes[k], key=lambda x: x.shipment_id)
    return dict(sorted(lanes.items(), key=lambda kv: kv[0]))


def pick_subset(items: List[ShipmentRow], pct: int) -> List[ShipmentRow]:
    if not items:
        return []
    pct = max(0, min(int(pct), 100))
    if pct >= 100:
        return items
    if pct <= 0:
        return []
    n = max(1, int(round(len(items) * (pct / 100.0))))
    return items[:n]


def post_simulate_scenario(
    api_base_url: str,
    *,
    shipment_ids: List[str],
    vehicle_type: Optional[str],
    consolidation_pct: int,
    edge_capacity: int,
    speed_kmph: float,
) -> Dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/simulate_scenario"

    vehicle_overrides = {sid: vehicle_type for sid in shipment_ids} if vehicle_type else None

    payload: Dict[str, Any] = {
        "shipment_ids": shipment_ids,
        "route_changes": {"optimize_with_engine": False, "overrides_by_shipment_id": None},
        "consolidation": {"enabled": bool(consolidation_pct > 0), "per_trip_overhead_emissions_kg": 0.0},
        "vehicle_type_changes": {"overrides_by_shipment_id": vehicle_overrides},
        "edge_capacity": int(edge_capacity),
        "speed_kmph": float(speed_kmph),
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def emissions_bar_chart(baseline_kg: float, simulated_kg: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(name="Baseline", x=["Emissions"], y=[baseline_kg]),
            go.Bar(name="Simulated", x=["Emissions"], y=[simulated_kg]),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=320,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="kg CO2e",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def render_digital_twin_section(api_base_url: str) -> None:
    st.subheader("Digital Twin Simulation")
    st.caption(
        "Simulate route/vehicle/consolidation scenarios before applying changes. "
        "Results are computed by the FastAPI backend via `POST /simulate_scenario`."
    )

    try:
        shipments = fetch_shipments(api_base_url)
    except Exception as e:
        st.error(f"Could not load shipments from backend. {e}")
        return

    if not shipments:
        st.info("No shipments available yet. Ingest shipments first.")
        return

    lanes = build_lane_index(shipments)
    lane_options = list(lanes.keys())

    with st.form("digital_twin_form", border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_lane = st.selectbox("Select route", lane_options, index=0)
        with col2:
            consolidation_pct = st.slider("Consolidation (%)", min_value=0, max_value=100, value=0, step=5)
        with col3:
            vehicle_type = st.selectbox(
                "Change vehicle type",
                options=["(no change)", "road", "truck_euro_6", "truck_euro_5", "rigid_18t", "articulated_40t"],
                index=0,
            )

        adv = st.expander("Advanced simulation settings")
        with adv:
            edge_capacity = st.number_input("Edge capacity (concurrency)", min_value=1, max_value=1000, value=5, step=1)
            speed_kmph = st.number_input("Speed (km/h)", min_value=1.0, max_value=120.0, value=60.0, step=5.0)

        submit = st.form_submit_button("Simulate", type="primary")

    lane_shipments = lanes.get(selected_lane, [])
    selected_shipments = pick_subset(lane_shipments, consolidation_pct if consolidation_pct > 0 else 100)
    selected_ids = [s.shipment_id for s in selected_shipments]

    st.write(
        f"Selected shipments: **{len(selected_ids)}** "
        f"(route has {len(lane_shipments)} total shipments; subset selection is based on the consolidation slider)."
    )

    if not submit:
        cached = st.session_state.get("digital_twin_last_result")
        if cached:
            _render_simulation_result(cached)
        return

    if not selected_ids:
        st.warning("No shipments selected for simulation. Increase consolidation % or pick a route with shipments.")
        return

    vt = None if vehicle_type == "(no change)" else vehicle_type

    with st.spinner("Running simulation..."):
        try:
            result = post_simulate_scenario(
                api_base_url,
                shipment_ids=selected_ids,
                vehicle_type=vt,
                consolidation_pct=int(consolidation_pct),
                edge_capacity=int(edge_capacity),
                speed_kmph=float(speed_kmph),
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = f" Response: {e.response.text}"
            except Exception:
                pass
            st.error(f"Simulation request failed. {e}{detail}")
            return
        except Exception as e:
            st.error(f"Simulation request failed. {e}")
            return

    st.session_state["digital_twin_last_result"] = result
    _render_simulation_result(result)


def _render_simulation_result(result: Dict[str, Any]) -> None:
    baseline = float(result.get("baseline_emissions_kg", 0.0))
    simulated = float(result.get("simulated_emissions_kg", 0.0))
    reduction_pct = float(result.get("emission_reduction_pct", 0.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline emissions (kg CO2e)", f"{baseline:,.2f}")
    c2.metric("Simulated emissions (kg CO2e)", f"{simulated:,.2f}")
    c3.metric("Reduction (%)", f"{reduction_pct:,.2f}")

    st.plotly_chart(emissions_bar_chart(baseline, simulated), use_container_width=True)

    with st.expander("Recommended changes (details)"):
        st.json(result.get("recommended_changes", {}))


def main() -> None:
    st.set_page_config(page_title="Carbon Tracker Dashboard", layout="wide")

    # ---------------- SIDEBAR ---------------- #
    with st.sidebar:
        st.title("🚛 Carbon Tracker")
        st.caption("AI Logistics Sustainability Platform")

        st.divider()

        st.subheader("🔗 Backend")
        api_base_url = st.text_input("API Base URL", value=_api_url_default())

        if st.button("🔄 Refresh Data"):
            fetch_shipments.clear()
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.subheader("📂 Navigation")

        # Initialize session state page
        if "page" not in st.session_state:
            st.session_state.page = "Digital Twin"

        # Sidebar Buttons
        if st.button("🧪 Digital Twin Simulation"):
            st.session_state.page = "Digital Twin"

        if st.button("🌍 Network Map"):
            st.session_state.page = "Network Map"

        if st.button("📊 Emissions Analytics"):
            st.session_state.page = "Emissions Analytics"

        if st.button("🌱 Carbon Recommendations"):
            st.session_state.page = "Carbon Recommendations"

        if st.button("🚀 Route Optimization"):
            st.session_state.page = "Route Optimization"

        if st.button("⚡ CO₂ Savings Simulator"):
            st.session_state.page = "CO2 Savings"

        if st.button("🧠 GNN Insights"):
            st.session_state.page = "GNN Insights"

        if st.button("📈 Emission Forecast"):
            st.session_state.page = "Emission Forecast"

    # ---------------- MAIN PAGE ---------------- #
    st.title("Logistics Carbon Tracker")

    page = st.session_state.page

    if page == "Digital Twin":
        render_digital_twin_section(api_base_url)

    elif page == "Network Map":
        render_network_map_section(api_base_url)

    elif page == "Emissions Analytics":
        render_emissions_analytics(api_base_url)

    elif page == "Carbon Recommendations":
        render_carbon_recommendations(api_base_url)

    elif page == "Route Optimization":
        render_route_optimization(api_base_url)

    elif page == "CO2 Savings":
        render_savings_simulator(api_base_url)

    elif page == "GNN Insights":
        render_gnn_results(api_base_url)

    elif page == "Emission Forecast":
        render_emission_forecast(api_base_url)
def render_network_map_section(api_base_url: str) -> None:
    import pandas as pd
    import pydeck as pdk

    st.header("🌍 Logistics Network Map")

    try:
        shipment_rows = fetch_shipments(api_base_url)
    except Exception as e:
         st.error(f"Could not fetch shipment data for map: {e}")
         return

    CITY_COORDS = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.7041, 77.1025),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Kolkata": (22.5726, 88.3639),
        "Ahmedabad": (23.0225, 72.5714),
    }

    rows = []
    for s in shipment_rows:
        origin = s.origin.strip().title()
        destination = s.destination.strip().title()

        if origin in CITY_COORDS and destination in CITY_COORDS:
            origin_lat, origin_lng = CITY_COORDS[origin]
            dest_lat, dest_lng = CITY_COORDS[destination]

            rows.append({
                "origin": origin,
                "destination": destination,
                "origin_lat": origin_lat,
                "origin_lng": origin_lng,
                "dest_lat": dest_lat,
                "dest_lng": dest_lng,
            })
    st.write("Rows on map:", len(rows))
    if not rows:
        st.info("No shipment data available to display on the map.")
        return

    df = pd.DataFrame(rows)
    arc_layer = pdk.Layer(
    "ArcLayer",
    data=df,
    get_source_position=["origin_lng", "origin_lat"],
    get_target_position=["dest_lng", "dest_lat"],
    get_source_color=[0, 255, 255, 200],   # bright cyan
    get_target_color=[255, 80, 0, 200],    # bright oran    ge
    get_width=5,                           # increase width
    width_scale=1,                         # IMPORTANT
    pickable=True,
    auto_highlight=True,
)
 

    view_state = pdk.ViewState(
    latitude=df["origin_lat"].mean(),
    longitude=df["origin_lng"].mean(),
    zoom=5,
    pitch=30,
)

    st.pydeck_chart(
        pdk.Deck(
            layers=[arc_layer],
            initial_view_state=view_state,
        ),
        use_container_width=True,
    )
def render_emissions_analytics(api_base_url: str) -> None:
    import pandas as pd

    st.header("📊 Emissions Analytics")

    try:
        shipment_rows = fetch_shipments(api_base_url)
    except Exception as e:
        st.error(f"Could not fetch shipment data: {e}")
        return

    if not shipment_rows:
        st.info("No shipment data available for analytics.")
        return

    # Convert dataclass list → DataFrame
    df = pd.DataFrame([{
        "shipment_id": s.shipment_id,
        "origin": s.origin,
        "destination": s.destination,
        "transport_mode": s.transport_mode,
        "co2e_kg": s.co2e_kg
    } for s in shipment_rows])
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", len(df))
    col2.metric("Total CO₂ Emissions", f"{df['co2e_kg'].sum():,.2f} kg")
    col3.metric("Avg Emissions per Shipment", f"{df['co2e_kg'].mean():,.2f} kg")
    city_filter = st.selectbox(
       "Filter by Origin City",
       ["All"] + sorted(df["origin"].unique().tolist())
    )

    if city_filter != "All":
        df = df[df["origin"] == city_filter]


    df = df.dropna(subset=["co2e_kg"])
    df["route"] = df["origin"] + " → " + df["destination"]
    top_routes = (
        df.groupby("route")["co2e_kg"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.download_button(
      "Download Emissions Report",
      df.to_csv(index=False),
      "carbon_emissions_report.csv",
      "text/csv",
      help="Download the current view as a CSV file"
    )
    st.subheader("Top 10 Highest Emission Routes")
    st.bar_chart(top_routes)
def render_carbon_recommendations(api_base_url: str) -> None:
    import pandas as pd

    st.header("🌱 Carbon Reduction Recommendations")

    try:
        shipment_rows = fetch_shipments(api_base_url)
    except Exception as e:
        st.error(f"Could not fetch shipment data: {e}")
        return

    if not shipment_rows:
        st.info("No shipment data available.")
        return

    # Convert shipments to DataFrame
    df = pd.DataFrame([{
        "shipment_id": s.shipment_id,
        "origin": s.origin,
        "destination": s.destination,
        "transport_mode": s.transport_mode,
        "co2e_kg": s.co2e_kg
    } for s in shipment_rows])

    df = df.dropna(subset=["co2e_kg"])

    # Create route column
    df["route"] = df["origin"] + " → " + df["destination"]

    # Identify highest emission routes
    route_emissions = (
        df.groupby(["route", "transport_mode"])["co2e_kg"]
        .sum()
        .reset_index()
        .sort_values("co2e_kg", ascending=False)
    )

    # Only recommend for diesel trucks
    recommendations = route_emissions[
        route_emissions["transport_mode"] == "diesel_truck"
    ].copy()

    if recommendations.empty:
        st.info("No diesel truck routes found for optimization.")
        return

    # Estimate potential reduction
    recommendations["recommended_vehicle"] = "electric_truck"
    recommendations["estimated_reduction_%"] = 25
    recommendations["potential_savings_kg"] = (
        recommendations["co2e_kg"] * 0.25
    )

    # Show top 10
    recommendations = recommendations.head(10)

    st.subheader("Top Routes for Emission Reduction")

    st.dataframe(
        recommendations[[
            "route",
            "transport_mode",
            "recommended_vehicle",
            "co2e_kg",
            "potential_savings_kg"
        ]]
        .rename(columns={
            "transport_mode": "Current Vehicle",
            "recommended_vehicle": "Suggested Vehicle",
            "co2e_kg": "Current Emissions (kg)",
            "potential_savings_kg": "Potential Savings (kg)"
        })
    )
def render_route_optimization(api_base_url: str) -> None:
    st.header("🚀 Route Optimization Demo")

    try:
        shipment_rows = fetch_shipments(api_base_url)
    except Exception as e:
        st.error(f"Failed to fetch shipments: {e}")
        return

    if not shipment_rows:
        st.warning("No shipments available.")
        return

    # Build route → shipment mapping
    routes = {}
    for s in shipment_rows:
        route = f"{s.origin} → {s.destination}"
        routes.setdefault(route, []).append(s.shipment_id)

    selected_route = st.selectbox("Select Route to Optimize", list(routes.keys()))

    shipment_ids = routes[selected_route]

    st.write(f"Shipments in this route: {len(shipment_ids)}")

    if st.button("Run Route Optimization"):

        payload = {
            "shipment_ids": shipment_ids,
            "alpha": 1,
            "beta": 1,
            "cost_per_km": 1,
            "max_route_candidates": 3,
            "per_trip_overhead_cost": 0,
            "per_trip_overhead_emissions_kg": 0
        }

        try:
            response = requests.post(
                f"{api_base_url}/optimize_routes",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            payload_json = response.json()
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            return

        # Backend returns {"result": {...}} where "result" contains a "totals" block.
        result = payload_json.get("result", {}) or {}
        totals = result.get("totals", {}) or {}

        baseline = float(totals.get("baseline_emissions_kg", result.get("baseline_emissions", 0.0)))
        optimized = float(totals.get("optimized_emissions_kg", 0.0))
        reduction = float(totals.get("emission_reduction_pct", 0.0))

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Emissions", f"{baseline:,.2f} kg")
        col2.metric("Optimized Emissions", f"{optimized:,.2f} kg")
        col3.metric("Reduction %", f"{reduction:.2f}%")

        st.success("Optimization Complete ✅")
def render_gnn_results(api_base_url: str) -> None:
    import pandas as pd

    st.header("🧠 GNN High Emission Route Detection")

    try:
        response = requests.get(f"{api_base_url}/high_emission_routes", timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Failed to fetch GNN results: {e}")
        return

    # API returns "anomalies" (list of {origin, destination, score, attributes})
    anomalies = data.get("anomalies", []) or []

    if not anomalies:
        st.info("No high emission routes detected.")
        return

    rows = [
        {"route": f"{a.get('origin', '')} → {a.get('destination', '')}", "anomaly_score": float(a.get("score", 0))}
        for a in anomalies
    ]
    df = pd.DataFrame(rows)

    st.subheader("🚨 Inefficient Routes (Anomaly Scores)")

    st.dataframe(df)

    if "anomaly_score" in df.columns and len(df) > 0:
        st.bar_chart(df.set_index("route")["anomaly_score"])
def render_emission_forecast(api_base_url: str) -> None:
    import pandas as pd

    st.header("📈 Emission Forecast")

    try:
        # Use small input_length/horizon: need 7+ points/lane; horizon=3 for a useful chart
        params = {"input_length": 4, "horizon": 3, "epochs": 30}
        response = requests.get(
            f"{api_base_url}/predict_emissions",
            params=params,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Forecast request failed: {e}. Response: {getattr(e.response, 'text', '')[:200]}")
        return
    except Exception as e:
        st.error(f"Failed to fetch forecast data: {e}")
        return

    # Backend returns: {"horizon": int, "predictions_by_lane": {lane_id: [values...]}}
    predictions_by_lane = data.get("predictions_by_lane", {}) or {}
    horizon = int(data.get("horizon", 0) or 0)

    if not predictions_by_lane or horizon <= 0:
        st.info("No forecast data available. Add more shipments (at least 3 per lane) to see predictions.")
        return

    # Aggregate predictions across lanes to get an average forecast per step.
    steps = list(range(1, horizon + 1))
    agg = [0.0] * horizon
    count = 0
    for series in predictions_by_lane.values():
        if not series:
            continue
        padded = list(series)[:horizon]
        if len(padded) < horizon:
            padded += [0.0] * (horizon - len(padded))
        agg = [a + float(b) for a, b in zip(agg, padded)]
        count += 1

    if count == 0:
        st.info("No forecast data available.")
        return

    avg = [v / count for v in agg]
    df = pd.DataFrame({"Forecast step": steps, "kg CO2e": avg})

    st.line_chart(df.set_index("Forecast step")["kg CO2e"])
def render_savings_simulator(api_base_url: str) -> None:
    st.header("⚡ CO₂ Savings Simulator")

    st.write("Estimate potential emission reductions by switching vehicle types.")

    current_emissions = st.number_input(
        "Current Emissions (kg CO₂)", 
        min_value=0.0, 
        value=1000.0
    )

    reduction_percent = st.slider(
        "Estimated Reduction (%)", 
        0, 
        100, 
        20
    )

    savings = current_emissions * (reduction_percent / 100)
    new_emissions = current_emissions - savings

    col1, col2 = st.columns(2)

    col1.metric("Potential Savings", f"{savings:.2f} kg")
    col2.metric("New Emissions", f"{new_emissions:.2f} kg")
if __name__ == "__main__":
    main()

