"""
Graph Neural Network models for logistics route intelligence.

This module provides:
- conversion of a NetworkX logistics graph into PyTorch Geometric format
- a GraphSAGE-based GNN model for lane intelligence
- training helpers
- utilities to generate route embeddings and detect inefficient routes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from app.services.graph_builder import (
    ATTR_AVG_UTILIZATION,
    ATTR_CARBON_INTENSITY,
    ATTR_DISTANCE,
    ATTR_SHIPMENT_FREQUENCY,
    ATTR_TOTAL_EMISSIONS,
    ATTR_TOTAL_TON_KM,
)


@dataclass
class GNNConfig:
    """
    Configuration for the route GNN.

    Attributes:
        hidden_channels: Hidden dimension for node and edge embeddings.
        num_layers: Number of GraphSAGE layers.
        dropout: Dropout probability between layers.
    """

    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.1


def networkx_to_pyg(graph: nx.DiGraph) -> Tuple[Data, Dict[int, Tuple[str, str]]]:
    """
    Convert a logistics NetworkX graph into PyTorch Geometric `Data`.

    Node features are simple structural features:
    - in_degree
    - out_degree
    - total_incident_shipments

    Edge features are taken from the graph attributes:
    - distance
    - total_emissions
    - shipment_frequency
    - average_vehicle_utilization
    - total_ton_km
    - carbon_intensity

    Args:
        graph: Directed NetworkX graph constructed by `graph_builder`.

    Returns:
        A tuple of:
        - `Data` object with `x`, `edge_index`, and `edge_attr`.
        - Mapping from edge index to (origin, destination) lane key.
    """
    # Map nodes to indices
    node_labels = list(graph.nodes())
    node_to_idx: Dict[str, int] = {label: i for i, label in enumerate(node_labels)}
    num_nodes = len(node_labels)

    # Node features
    x_list: List[List[float]] = []
    for node in node_labels:
        in_deg = float(graph.in_degree(node))
        out_deg = float(graph.out_degree(node))
        total_shipments = 0.0
        for _, _, data in graph.in_edges(node, data=True):
            total_shipments += float(data.get(ATTR_SHIPMENT_FREQUENCY, 0.0))
        for _, _, data in graph.out_edges(node, data=True):
            total_shipments += float(data.get(ATTR_SHIPMENT_FREQUENCY, 0.0))
        x_list.append([in_deg, out_deg, total_shipments])

    x = torch.tensor(x_list, dtype=torch.float32)

    # Edge index and attributes
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    edge_attr_list: List[List[float]] = []
    edge_index_to_lane: Dict[int, Tuple[str, str]] = {}

    for edge_idx, (u, v, data) in enumerate(graph.edges(data=True)):
        edge_sources.append(node_to_idx[u])
        edge_targets.append(node_to_idx[v])
        edge_attr_list.append(
            [
                float(data.get(ATTR_DISTANCE, 0.0)),
                float(data.get(ATTR_TOTAL_EMISSIONS, 0.0)),
                float(data.get(ATTR_SHIPMENT_FREQUENCY, 0.0)),
                float(data.get(ATTR_AVG_UTILIZATION, 0.0)),
                float(data.get(ATTR_TOTAL_TON_KM, 0.0)),
                float(data.get(ATTR_CARBON_INTENSITY, 0.0)),
            ]
        )
        edge_index_to_lane[edge_idx] = (u, v)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    return data, edge_index_to_lane


class RouteGNN(nn.Module):
    """
    GraphSAGE-based GNN for route intelligence.

    Produces node embeddings and edge-level scores/embeddings
    for logistics lanes (shipment routes).
    """

    def __init__(self, in_node_channels: int, in_edge_channels: int, config: GNNConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GNNConfig()
        self.config = config

        hidden = config.hidden_channels

        # Initial encoders
        self.node_encoder = nn.Linear(in_node_channels, hidden)
        self.edge_encoder = nn.Linear(in_edge_channels, hidden)

        # GraphSAGE layers
        self.convs = nn.ModuleList(
            [SAGEConv(hidden, hidden) for _ in range(config.num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()

        # Edge embedding and scoring heads
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.score_head = nn.Linear(hidden, 1)

    def encode_nodes(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Encode node features into node embeddings.

        Args:
            x: Node feature matrix [num_nodes, in_node_channels].
            edge_index: Edge index tensor [2, num_edges].

        Returns:
            Node embeddings [num_nodes, hidden_channels].
        """
        h = self.activation(self.node_encoder(x))
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def encode_edges(self, node_emb: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Compute edge embeddings from node embeddings and edge attributes.

        Args:
            node_emb: Node embeddings [num_nodes, hidden_channels].
            edge_index: Edge index [2, num_edges].
            edge_attr: Edge features [num_edges, in_edge_channels].

        Returns:
            Edge embeddings [num_edges, hidden_channels].
        """
        edge_emb_in = self.edge_encoder(edge_attr)
        src, dst = edge_index
        h_src = node_emb[src]
        h_dst = node_emb[dst]
        concat = torch.cat([h_src, h_dst, edge_emb_in], dim=-1)
        edge_emb = self.edge_mlp(concat)
        return edge_emb

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass that returns edge-level scores.

        Args:
            data: PyG Data object with x, edge_index, edge_attr.

        Returns:
            Edge scores tensor [num_edges, 1], where higher values
            can correspond to more inefficient or higher-emission lanes.
        """
        node_emb = self.encode_nodes(data.x, data.edge_index)
        edge_emb = self.encode_edges(node_emb, data.edge_index, data.edge_attr)
        scores = self.score_head(edge_emb)
        return scores


def train_route_gnn(
    model: RouteGNN,
    data: Data,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device | None = None,
) -> RouteGNN:
    """
    Train the route GNN in a supervised or self-supervised regression regime.

    By default, the training target is the carbon intensity of each edge,
    encouraging the model to learn structural patterns related to emission intensity.

    Args:
        model: RouteGNN instance.
        data: PyG Data object with x, edge_index, edge_attr.
        num_epochs: Number of training epochs.
        optimizer: Optimizer instance (e.g., Adam).
        device: Optional device to train on.

    Returns:
        The trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    data = data.to(device)
    criterion = nn.MSELoss()

    # Carbon intensity is the last column in our edge_attr vector
    target = data.edge_attr[:, -1].unsqueeze(-1)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, target)
        loss.backward()
        optimizer.step()

    return model


def generate_route_embeddings(
    model: RouteGNN,
    data: Data,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate route (edge) embeddings from a trained model.

    Args:
        model: Trained RouteGNN.
        data: PyG Data object.
        device: Optional device for inference.

    Returns:
        Edge embeddings tensor of shape [num_edges, hidden_channels].
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        node_emb = model.encode_nodes(data.x, data.edge_index)
        edge_emb = model.encode_edges(node_emb, data.edge_index, data.edge_attr)
    return edge_emb


def detect_inefficient_lanes(
    graph: nx.DiGraph,
    model: RouteGNN,
    z_threshold: float = 2.0,
    device: torch.device | None = None,
) -> List[Dict[str, object]]:
    """
    Detect inefficient or high-emission lanes using the trained GNN.

    The model produces an edge-level score per lane. Edges with scores
    greater than mean + z_threshold * std are flagged as anomalies.

    Args:
        graph: Logistics NetworkX DiGraph from `graph_builder`.
        model: Trained RouteGNN model.
        z_threshold: Z-score threshold for flagging anomalies.
        device: Optional device for inference.

    Returns:
        A list of dictionaries describing anomalous lanes, including:
        - origin
        - destination
        - score
        - attributes (original edge attributes)
    """
    data, edge_index_to_lane = networkx_to_pyg(graph)

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        scores = model(data).squeeze(-1).cpu()

    mean = scores.mean().item()
    std = scores.std(unbiased=False).item() if scores.numel() > 1 else 0.0
    threshold = mean + z_threshold * std if std > 0 else mean

    anomalies: List[Dict[str, object]] = []
    for edge_idx, score in enumerate(scores.tolist()):
        origin, dest = edge_index_to_lane[edge_idx]
        attrs = graph.edges[origin, dest]
        is_anomalous = score >= threshold
        attrs["gnn_score"] = score
        attrs["gnn_is_anomalous"] = is_anomalous

        if is_anomalous:
            anomalies.append(
                {
                    "origin": origin,
                    "destination": dest,
                    "score": score,
                    "attributes": dict(attrs),
                }
            )

    return anomalies

