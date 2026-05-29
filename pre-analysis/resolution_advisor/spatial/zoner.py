"""
Spatial zone generator -- backends for gridflow and PyPSA clustering.
Currently stubs; enable by setting ENABLED = True and installing dependencies.
"""
from __future__ import annotations

GRIDFLOW_ENABLED = False
PYPSA_ENABLED = False


def partition_gridflow(countries: list[str], n_zones: int, datasets_path: str):
    """
    Partition a region into n_zones using ESMAP gridflow.

    Requires:  pip install gridflow  (private ESMAP repo)
    Reference: https://github.com/ESMAP-World-Bank-Group/gridflow
    """
    if not GRIDFLOW_ENABLED:
        raise NotImplementedError(
            "gridflow backend not enabled. "
            "Set GRIDFLOW_ENABLED = True and install gridflow."
        )
    from gridflow import model  # noqa: F401
    mod = model.region(countries, datasets_path)
    mod.create_zones(n=n_zones)
    mod.create_network()
    return mod


def partition_pypsa(network, n_zones: int, weighting: str = "load"):
    """
    Cluster a PyPSA network into n_zones using k-means on substations.

    Requires:  pip install pypsa
    Inspired by: PyPSA-Eur cluster_network.py
    Reference: Horsch & Brown 2017 (arXiv:1705.07617)
    Weighting options: 'load', 'generation', 'uniform'
    """
    if not PYPSA_ENABLED:
        raise NotImplementedError(
            "PyPSA clustering backend not enabled. "
            "Set PYPSA_ENABLED = True and install pypsa."
        )
    from pypsa.clustering.spatial import busmap_by_kmeans  # noqa: F401
    busmap = busmap_by_kmeans(network, weighting=weighting, n_clusters=n_zones)
    return busmap
