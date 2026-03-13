"""
Graph builder for the logistics network.

Constructs a NetworkX directed graph from shipment data with:
- Nodes: locations (origin and destination cities)
- Edges: shipment lanes with distance, emissions, frequency, and utilization
"""

from collections import defaultdict
from typing import Any, Iterable

import networkx as nx

from app.models.shipment_model import Shipment
from app.utils.emission_factors import get_vehicle_profile

# Edge attribute keys
ATTR_DISTANCE = "distance"
ATTR_TOTAL_EMISSIONS = "total_emissions"
ATTR_SHIPMENT_FREQUENCY = "shipment_frequency"
ATTR_AVG_UTILIZATION = "average_vehicle_utilization"
ATTR_TOTAL_TON_KM = "total_ton_km"
ATTR_CARBON_INTENSITY = "carbon_intensity"


def build_graph_from_shipments(shipments: Iterable[Shipment]) -> nx.DiGraph:
    """
    Build a directed logistics network graph from shipment records.

    Nodes represent locations (origin and destination cities).
    Edges represent shipment lanes with attributes:
    - distance: average distance in km
    - total_emissions: sum of CO2e in kg
    - shipment_frequency: number of shipments on the lane
    - average_vehicle_utilization: mean of (weight / capacity) across shipments

    Args:
        shipments: Iterable of Shipment records with co2e_kg computed.

    Returns:
        NetworkX DiGraph with nodes and edges populated.
    """
    graph = nx.DiGraph()

    # Accumulate: (total_dist, total_emissions, count, util_sum, total_ton_km)
    lane_data: dict[tuple[str, str], tuple[float, float, int, float, float]] = defaultdict(
        lambda: (0.0, 0.0, 0, 0.0, 0.0)
    )

    for shipment in shipments:
        origin = shipment.origin_location
        dest = shipment.destination_location

        graph.add_node(origin)
        graph.add_node(dest)

        profile = get_vehicle_profile(shipment.transport_mode)
        utilization = shipment.weight_tons / profile.capacity_tons
        co2e = shipment.co2e_kg or 0.0
        ton_km = shipment.distance_km * shipment.weight_tons

        prev = lane_data[(origin, dest)]
        lane_data[(origin, dest)] = (
            prev[0] + shipment.distance_km,
            prev[1] + co2e,
            prev[2] + 1,
            prev[3] + utilization,
            prev[4] + ton_km,
        )

    for (origin, dest), (total_dist, total_emissions, count, util_sum, total_ton_km) in lane_data.items():
        avg_distance = total_dist / count if count > 0 else 0.0
        avg_utilization = util_sum / count if count > 0 else 0.0

        graph.add_edge(
            origin,
            dest,
            **{
                ATTR_DISTANCE: avg_distance,
                ATTR_TOTAL_EMISSIONS: total_emissions,
                ATTR_SHIPMENT_FREQUENCY: count,
                ATTR_AVG_UTILIZATION: avg_utilization,
                ATTR_TOTAL_TON_KM: total_ton_km,
            },
        )

    return graph


def compute_lane_carbon_intensity(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute lane-level carbon intensity and add it as an edge attribute.

    Carbon intensity = total_emissions / total_ton_km (kg CO2e per ton-km).
    This is the standard metric for freight emissions intensity.

    Args:
        graph: DiGraph with total_emissions and total_ton_km on edges.

    Returns:
        The same graph, modified in place with carbon_intensity on each edge.
    """
    for u, v, data in graph.edges(data=True):
        total_emissions = data.get(ATTR_TOTAL_EMISSIONS, 0.0)
        total_ton_km = data.get(ATTR_TOTAL_TON_KM, 0.0)

        if total_ton_km > 0:
            data[ATTR_CARBON_INTENSITY] = total_emissions / total_ton_km
        else:
            data[ATTR_CARBON_INTENSITY] = 0.0

    return graph


def export_for_ml(graph: nx.DiGraph) -> dict[str, Any]:
    """
    Export the graph structure for machine learning models (e.g., PyTorch Geometric).

    Returns a dictionary with:
    - node_to_idx: mapping from node label to integer index
    - edge_index: 2xE array of [source_indices, target_indices]
    - edge_features: list of edge attribute vectors [distance, total_emissions,
      shipment_frequency, average_vehicle_utilization, total_ton_km, carbon_intensity]
    - node_labels: ordered list of node names for reverse lookup

    Args:
        graph: DiGraph with lane attributes (from build_graph_from_shipments
               and compute_lane_carbon_intensity).

    Returns:
        Dictionary suitable for conversion to tensors in ML pipelines.
    """
    node_labels = list(graph.nodes())
    node_to_idx = {label: i for i, label in enumerate(node_labels)}

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_features: list[list[float]] = []

    for u, v, data in graph.edges(data=True):
        edge_sources.append(node_to_idx[u])
        edge_targets.append(node_to_idx[v])
        edge_features.append(
            [
                data.get(ATTR_DISTANCE, 0.0),
                data.get(ATTR_TOTAL_EMISSIONS, 0.0),
                float(data.get(ATTR_SHIPMENT_FREQUENCY, 0)),
                data.get(ATTR_AVG_UTILIZATION, 0.0),
                data.get(ATTR_TOTAL_TON_KM, 0.0),
                data.get(ATTR_CARBON_INTENSITY, 0.0),
            ]
        )

    return {
        "node_to_idx": node_to_idx,
        "node_labels": node_labels,
        "edge_index": [edge_sources, edge_targets],
        "edge_features": edge_features,
        "num_nodes": len(node_labels),
        "num_edges": len(edge_sources),
    }


class GraphBuilder:
    """
    Builder for logistics network graphs.

    Provides a stateful interface that wraps the module-level functions.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph | None = None

    def build_from_shipments(self, shipments: Iterable[Shipment]) -> nx.DiGraph:
        """
        Build the graph from shipment records and store it.

        Returns the constructed DiGraph.
        """
        self._graph = build_graph_from_shipments(shipments)
        return self._graph

    def compute_carbon_intensity(self) -> nx.DiGraph:
        """
        Compute lane-level carbon intensity on the current graph.

        Raises:
            ValueError: If no graph has been built yet.
        """
        if self._graph is None:
            raise ValueError("No graph built yet. Call build_from_shipments first.")
        return compute_lane_carbon_intensity(self._graph)

    def export_for_ml(self) -> dict[str, Any]:
        """
        Export the current graph for ML models.

        Raises:
            ValueError: If no graph has been built yet.
        """
        if self._graph is None:
            raise ValueError("No graph built yet. Call build_from_shipments first.")
        return export_for_ml(self._graph)

    @property
    def graph(self) -> nx.DiGraph | None:
        """The current graph, or None if not yet built."""
        return self._graph
