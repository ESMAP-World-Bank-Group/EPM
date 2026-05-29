"""
Estimate number of network bottleneck corridors from OSM HV line topology.

Method (inspired by PyPSA-Eur):
  1. Build an undirected graph: HV lines as edges, substations as nodes
     (snapped to nearest substation within tolerance)
  2. Compute edge betweenness centrality -- lines that many shortest paths
     pass through are likely bottlenecks
  3. Lines with betweenness > mean + 2*std are flagged as bottleneck corridors
  4. Cluster adjacent bottleneck lines -> count distinct corridors

Returns (n_bottlenecks, coverage_note).
Falls back to 0 if NetworkX is not installed or OSM data is too sparse.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Tuple


SNAP_TOLERANCE_KM = 15    # substations within this distance are merged
MIN_LINES_FOR_ANALYSIS = 5  # below this, graph is too sparse to be meaningful


def compute_network_bottlenecks(
    country_iso: str,
    substations: List[dict],
    hv_lines: List[dict],
    boundary_gdf=None,
) -> Tuple[int, str]:
    """
    Returns (n_bottleneck_corridors, note).
    """
    try:
        import networkx as nx
    except ImportError:
        return 0, "NetworkX not installed -- install with: pip install networkx"

    if len(hv_lines) < MIN_LINES_FOR_ANALYSIS:
        coverage = "sparse" if len(hv_lines) > 0 else "none"
        return 0, (
            f"OSM coverage too sparse ({len(hv_lines)} HV lines found) "
            f"for {country_iso} -- bottleneck analysis skipped"
        )

    # Build graph from lines
    G = _build_graph(substations, hv_lines)

    if G.number_of_nodes() < 4 or G.number_of_edges() < MIN_LINES_FOR_ANALYSIS:
        return 0, (
            f"Graph too small ({G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges) -- cannot detect bottlenecks"
        )

    # Edge betweenness centrality
    try:
        betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    except Exception as e:
        return 0, f"Betweenness computation failed: {e}"

    if not betweenness:
        return 0, "No betweenness values computed"

    values = list(betweenness.values())
    mean_b = sum(values) / len(values)
    std_b = math.sqrt(sum((v - mean_b) ** 2 for v in values) / len(values))
    threshold = mean_b + 2 * std_b

    bottleneck_edges = [e for e, b in betweenness.items() if b > threshold]

    # Cluster adjacent bottleneck edges into distinct corridors
    n_corridors = _count_corridors(bottleneck_edges)

    note = (
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges from OSM; "
        f"{len(bottleneck_edges)} bottleneck edges -> {n_corridors} corridor(s) "
        f"(betweenness threshold: {threshold:.4f})"
    )
    return n_corridors, note


# -- graph building ------------------------------------------------------------

def _build_graph(substations: List[dict], hv_lines: List[dict]):
    """
    Build a NetworkX graph from HV line geometry.
    Line endpoints are snapped to nearby substations.
    """
    try:
        import networkx as nx
    except ImportError:
        raise

    G = nx.Graph()

    # Index substations for fast lookup
    sub_points = [(s["lat"], s["lon"], s["osm_id"]) for s in substations if "lat" in s]

    for line in hv_lines:
        coords = line.get("coords", [])
        if len(coords) < 2:
            continue
        # Use first and last coordinate as endpoints
        start = (coords[0][1], coords[0][0])   # (lat, lon)
        end = (coords[-1][1], coords[-1][0])

        node_a = _snap_to_substation(start, sub_points) or f"pt_{start[0]:.3f}_{start[1]:.3f}"
        node_b = _snap_to_substation(end, sub_points) or f"pt_{end[0]:.3f}_{end[1]:.3f}"

        if node_a == node_b:
            continue

        G.add_node(node_a, lat=start[0], lon=start[1])
        G.add_node(node_b, lat=end[0], lon=end[1])
        G.add_edge(node_a, node_b,
                   voltage_kv=line.get("voltage_kv", 0),
                   osm_id=line.get("osm_id"))

    return G


def _snap_to_substation(
    point: Tuple[float, float],
    substations: List[Tuple[float, float, str]],
    tolerance_km: float = SNAP_TOLERANCE_KM,
) -> str | None:
    """Find the nearest substation within tolerance, return its osm_id."""
    best_id, best_dist = None, float("inf")
    for slat, slon, sid in substations:
        d = _haversine_km(point, (slat, slon))
        if d < best_dist:
            best_dist = d
            best_id = str(sid)
    return best_id if best_dist <= tolerance_km else None


def _count_corridors(bottleneck_edges: list) -> int:
    """Count distinct connected components among bottleneck edges."""
    if not bottleneck_edges:
        return 0
    adj = defaultdict(set)
    for a, b in bottleneck_edges:
        adj[a].add(b)
        adj[b].add(a)
    visited, n = set(), 0
    for node in adj:
        if node not in visited:
            n += 1
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    stack.extend(adj[cur] - visited)
    return n


def _haversine_km(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
