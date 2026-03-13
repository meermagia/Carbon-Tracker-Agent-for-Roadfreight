"""
Carbon heatmap visualization for logistics routes.

This module turns per-route CO2e outputs into geospatial artifacts (GeoJSON) and
an interactive Plotly visualization over OpenStreetMap tiles.

Key design choices:
- No Mapbox token: uses Plotly mapbox layers with style="open-street-map".
- "Heatmap" intensity is created by sampling points along each route weighted
  by route emissions, then rendering a density layer.
- Route lines and midpoint markers provide clear per-route tooltips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True, slots=True)
class RouteEmission:
    """
    A logistics route with geographic endpoints and associated emissions.

    Args:
        origin_lat: Origin latitude in degrees.
        origin_lon: Origin longitude in degrees.
        destination_lat: Destination latitude in degrees.
        destination_lon: Destination longitude in degrees.
        emissions_kg: Route emissions in kg CO2e (non-negative).
        route_id: Optional identifier for display/trace labeling.
    """

    origin_lat: float
    origin_lon: float
    destination_lat: float
    destination_lon: float
    emissions_kg: float
    route_id: str | None = None


def _validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")


def _validate_route(route: RouteEmission) -> None:
    _validate_lat_lon(route.origin_lat, route.origin_lon)
    _validate_lat_lon(route.destination_lat, route.destination_lon)
    if float(route.emissions_kg) < 0:
        raise ValueError(f"Emissions must be non-negative (kg CO2e). Got: {route.emissions_kg}")


def _as_route_emission(obj: RouteEmission | Mapping[str, Any]) -> RouteEmission:
    if isinstance(obj, RouteEmission):
        return obj

    emissions = obj.get("emissions_kg", obj.get("co2e_kg", obj.get("emissions")))
    if emissions is None:
        raise KeyError("Route missing emissions. Expected one of: emissions_kg, co2e_kg, emissions")

    return RouteEmission(
        origin_lat=float(obj["origin_lat"]),
        origin_lon=float(obj["origin_lon"]),
        destination_lat=float(obj["destination_lat"]),
        destination_lon=float(obj["destination_lon"]),
        emissions_kg=float(emissions),
        route_id=(str(obj["route_id"]) if "route_id" in obj and obj["route_id"] is not None else None),
    )


def convert_routes_to_geojson(
    routes: Iterable[RouteEmission | Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Convert route emissions to a GeoJSON FeatureCollection of LineStrings.

    This is useful for interoperability with GIS tools and for storing/rendering
    in other map frameworks.
    """
    features: list[dict[str, Any]] = []
    for r_in in routes:
        r = _as_route_emission(r_in)
        _validate_route(r)

        props: dict[str, Any] = {"emissions_kg": float(r.emissions_kg)}
        if r.route_id is not None:
            props["route_id"] = r.route_id

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(r.origin_lon), float(r.origin_lat)],
                        [float(r.destination_lon), float(r.destination_lat)],
                    ],
                },
                "properties": props,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def build_routes_from_shipments(
    shipments: Iterable[Any],
    *,
    location_to_coords: Mapping[str, tuple[float, float]],
    emissions_attr: str = "co2e_kg",
    origin_attr: str = "origin_location",
    destination_attr: str = "destination_location",
    route_id_attr: str = "shipment_id",
) -> list[RouteEmission]:
    """
    Build `RouteEmission` records from Shipment-like objects.

    This integrates with the carbon emission engine output (which stores `co2e_kg`
    on `Shipment`) by mapping origin/destination location labels to coordinates.

    Args:
        shipments: Iterable of objects (e.g., `Shipment`) with origin/destination
            fields and an emissions field (default: `co2e_kg`).
        location_to_coords: Map of location label -> (lat, lon).
        emissions_attr: Attribute name for emissions on each shipment.
        origin_attr: Attribute name for origin location label.
        destination_attr: Attribute name for destination location label.
        route_id_attr: Attribute used as a per-route id if present.

    Returns:
        List of `RouteEmission` (shipments with missing coords or emissions are skipped).
    """
    out: list[RouteEmission] = []
    for s in shipments:
        origin = getattr(s, origin_attr, None)
        dest = getattr(s, destination_attr, None)
        if not origin or not dest:
            continue

        if origin not in location_to_coords or dest not in location_to_coords:
            continue

        emissions = getattr(s, emissions_attr, None)
        if emissions is None:
            continue

        origin_lat, origin_lon = location_to_coords[origin]
        dest_lat, dest_lon = location_to_coords[dest]
        route_id = getattr(s, route_id_attr, None)

        r = RouteEmission(
            origin_lat=float(origin_lat),
            origin_lon=float(origin_lon),
            destination_lat=float(dest_lat),
            destination_lon=float(dest_lon),
            emissions_kg=float(emissions),
            route_id=(str(route_id) if route_id is not None else None),
        )
        _validate_route(r)
        out.append(r)
    return out


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [float(a)]
    step = (float(b) - float(a)) / (n - 1)
    return [float(a) + i * step for i in range(n)]


def _sample_route_points(
    route: RouteEmission,
    *,
    points_per_route: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Sample points along a route line for density rendering.

    Returns (lats, lons, weights).
    """
    lats = _linspace(route.origin_lat, route.destination_lat, points_per_route)
    lons = _linspace(route.origin_lon, route.destination_lon, points_per_route)
    w = float(route.emissions_kg)
    weights = [w] * len(lats)
    return lats, lons, weights


def generate_carbon_heatmap(
    routes: Sequence[RouteEmission | Mapping[str, Any]],
    *,
    title: str = "Logistics Carbon Heatmap",
    center: tuple[float, float] | None = None,
    zoom: float | None = None,
    points_per_route: int = 25,
    heatmap_radius: int = 18,
    heatmap_opacity: float = 0.55,
    colorscale: str = "YlOrRd",
    line_width_range: tuple[float, float] = (1.5, 7.0),
) -> Any:
    """
    Generate an interactive carbon heatmap of logistics routes.

    The visualization includes:
    - Density heatmap intensity (emissions-weighted samples along routes)
    - Route polylines colored by emissions
    - Midpoint markers with tooltips showing emissions

    Args:
        routes: Sequence of routes as `RouteEmission` or dict-like objects with
            origin/destination lat/lon and emissions.
        center: Optional (lat, lon). If omitted, computed from route midpoints.
        zoom: Optional initial zoom. If omitted, chosen based on route spread.
        points_per_route: Number of sampled points per route for the density layer.
        heatmap_radius: Pixel radius for the density kernel.
        heatmap_opacity: Opacity of the density layer.
        colorscale: Plotly colorscale name.
        line_width_range: (min_width, max_width) mapped from emission intensity.

    Returns:
        A Plotly Figure.
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Plotly is required for carbon heatmap visualization. "
            "Install it with: pip install plotly"
        ) from e

    if not routes:
        raise ValueError("No routes provided.")

    parsed: list[RouteEmission] = []
    for r in routes:
        rr = _as_route_emission(r)
        _validate_route(rr)
        parsed.append(rr)

    emissions = [float(r.emissions_kg) for r in parsed]
    e_min = min(emissions)
    e_max = max(emissions)
    denom = (e_max - e_min) if e_max != e_min else 1.0

    # Sample points for heatmap layer
    hm_lat: list[float] = []
    hm_lon: list[float] = []
    hm_z: list[float] = []
    for r in parsed:
        lats, lons, weights = _sample_route_points(r, points_per_route=max(2, int(points_per_route)))
        hm_lat.extend(lats)
        hm_lon.extend(lons)
        hm_z.extend(weights)

    # Midpoints for tooltips + centering
    mid_lat = [(r.origin_lat + r.destination_lat) / 2.0 for r in parsed]
    mid_lon = [(r.origin_lon + r.destination_lon) / 2.0 for r in parsed]

    if center is None:
        center = (sum(mid_lat) / len(mid_lat), sum(mid_lon) / len(mid_lon))

    if zoom is None:
        # Rough heuristic: bigger bounding box => smaller zoom
        lat_span = max([r.origin_lat for r in parsed] + [r.destination_lat for r in parsed]) - min(
            [r.origin_lat for r in parsed] + [r.destination_lat for r in parsed]
        )
        lon_span = max([r.origin_lon for r in parsed] + [r.destination_lon for r in parsed]) - min(
            [r.origin_lon for r in parsed] + [r.destination_lon for r in parsed]
        )
        span = max(lat_span, lon_span)
        if span > 60:
            zoom = 1.6
        elif span > 30:
            zoom = 2.4
        elif span > 15:
            zoom = 3.4
        elif span > 8:
            zoom = 4.5
        else:
            zoom = 5.8

    fig = go.Figure()

    # Emissions intensity layer
    fig.add_trace(
        go.Densitymapbox(
            lat=hm_lat,
            lon=hm_lon,
            z=hm_z,
            radius=int(heatmap_radius),
            opacity=float(heatmap_opacity),
            colorscale=colorscale,
            hoverinfo="skip",
            showscale=False,
            name="Emission intensity",
        )
    )

    # Route lines (per-route traces to support per-route coloring + hover)
    from plotly.colors import sample_colorscale  # local import keeps optional dependency isolated

    w_min, w_max = float(line_width_range[0]), float(line_width_range[1])
    for r in parsed:
        t = (float(r.emissions_kg) - e_min) / denom
        width = w_min + t * (w_max - w_min)
        # Scattermapbox line.color expects a CSS color, not a numeric value.
        route_color = sample_colorscale(colorscale, t)[0]
        label = r.route_id or "route"
        fig.add_trace(
            go.Scattermapbox(
                lat=[r.origin_lat, r.destination_lat],
                lon=[r.origin_lon, r.destination_lon],
                mode="lines",
                line={"width": width, "color": route_color},
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "CO2e: %{customdata:.2f} kg<extra></extra>"
                ),
                customdata=[float(r.emissions_kg), float(r.emissions_kg)],
                name="Routes",
                showlegend=False,
            )
        )

    # Midpoint markers for tooltips + colorbar
    custom = [{"emissions_kg": float(r.emissions_kg), "route_id": r.route_id} for r in parsed]
    fig.add_trace(
        go.Scattermapbox(
            lat=mid_lat,
            lon=mid_lon,
            mode="markers",
            marker={
                "size": 9,
                "opacity": 0.9,
                "color": emissions,
                "colorscale": colorscale,
                "cmin": float(e_min),
                "cmax": float(e_max),
                "colorbar": {"title": "CO2e (kg)"},
            },
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata.route_id}</b><br>"
                "CO2e: %{customdata.emissions_kg:.2f} kg<extra></extra>"
            ),
            name="Route emissions",
        )
    )

    fig.update_layout(
        title=title,
        mapbox={
            "style": "open-street-map",
            "center": {"lat": float(center[0]), "lon": float(center[1])},
            "zoom": float(zoom),
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    return fig


def geojson_to_plotly_sources(geojson: Mapping[str, Any]) -> list[RouteEmission]:
    """
    Convert a GeoJSON FeatureCollection (as produced by `convert_routes_to_geojson`)
    back to `RouteEmission` objects.
    """
    features = geojson.get("features", [])
    out: list[RouteEmission] = []
    for f in features:
        geom = f.get("geometry") or {}
        props: MutableMapping[str, Any] = dict(f.get("properties") or {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        (o_lon, o_lat), (d_lon, d_lat) = coords[0], coords[-1]
        emissions = props.get("emissions_kg", props.get("co2e_kg", props.get("emissions")))
        if emissions is None:
            continue
        out.append(
            RouteEmission(
                origin_lat=float(o_lat),
                origin_lon=float(o_lon),
                destination_lat=float(d_lat),
                destination_lon=float(d_lon),
                emissions_kg=float(emissions),
                route_id=(str(props["route_id"]) if "route_id" in props and props["route_id"] is not None else None),
            )
        )
    return out

