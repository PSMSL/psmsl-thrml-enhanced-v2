"""
PSMSL (Projected Symmetry Mirrored Semantic Lattice) models for THRML.

This module provides geometric energy-based models that incorporate:
- Mirror symmetry constraints
- Phi-scaling (golden ratio) indexing
- Colocated data and latent planes
- Spatial coherence through pairwise couplings
"""

from typing import Dict, List, Tuple, Optional
import jax.numpy as jnp
import networkx as nx
from ..pgm import AbstractNode
from .discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from ..block_management import Block


class DataSpin(AbstractNode):
    """Node representing observable data in PSMSL model."""
    pass


class LatentSpin(AbstractNode):
    """Node representing latent/hidden structure in PSMSL model."""
    pass


class PSMSLConfig:
    """Configuration for PSMSL energy-based model."""
    
    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        j_local: float = 0.6,
        j_latent: float = 0.4,
        j_dyad: float = 0.8,
        j_mirror: float = 0.25,
        h_data: float = 0.0,
        h_latent: float = 0.05,
    ):
        """
        Initialize PSMSL configuration.
        
        Args:
            rows: Number of rows in grid
            cols: Number of columns in grid
            j_local: Coupling strength for data-to-data neighbors
            j_latent: Coupling strength for latent-to-latent neighbors
            j_dyad: Coupling strength for data-to-latent (vertical) connections
            j_mirror: Coupling strength for mirror symmetry constraints
            h_data: Bias for data plane nodes
            h_latent: Bias for latent plane nodes
        """
        self.rows = rows
        self.cols = cols
        self.j_local = j_local
        self.j_latent = j_latent
        self.j_dyad = j_dyad
        self.j_mirror = j_mirror
        self.h_data = h_data
        self.h_latent = h_latent


def make_grid(rows: int, cols: int) -> nx.Graph:
    """Create a 2D grid graph."""
    return nx.grid_graph(dim=(rows, cols), periodic=False)


def relabel_grid_to_nodes(G: nx.Graph, node_cls):
    """Relabel grid coordinates to node objects."""
    nodes = {(r, c): node_cls() for (r, c) in G.nodes()}
    nx.relabel_nodes(G, nodes, copy=False)
    for coord, node in nodes.items():
        G.nodes[node]["coords"] = coord
    return G


def bipartite_blocks(G: nx.Graph) -> Tuple[List[AbstractNode], List[AbstractNode]]:
    """Create bipartite coloring for parallel Gibbs sampling."""
    coloring = nx.bipartite.color(G)
    color0 = [n for n, c in coloring.items() if c == 0]
    color1 = [n for n, c in coloring.items() if c == 1]
    return color0, color1


def build_phi_mirror_pairs(Gd: nx.Graph, mode: str = "simple") -> Dict:
    """
    Build mirror pairs based on phi-scaling or symmetry.
    
    Args:
        Gd: Data plane graph
        mode: "simple" for column pooling, "phi" for golden ratio scaling,
              "reflect" for reflection symmetry
    
    Returns:
        Dictionary mapping nodes to their mirror partners
    """
    pairs = {}
    
    if mode == "simple":
        # Simple column pooling (placeholder)
        for node in Gd.nodes():
            r, c = Gd.nodes[node]["coords"]
            target = (r, c // 2)
            for n2 in Gd.nodes():
                if Gd.nodes[n2]["coords"] == target and n2 != node:
                    pairs[node] = n2
                    break
    
    elif mode == "phi":
        # Phi-scaling: map to position scaled by golden ratio
        phi = (1 + jnp.sqrt(5)) / 2
        for node in Gd.nodes():
            r, c = Gd.nodes[node]["coords"]
            # Scale column by phi and round
            target_c = int(c / phi)
            target = (r, target_c)
            for n2 in Gd.nodes():
                if Gd.nodes[n2]["coords"] == target and n2 != node:
                    pairs[node] = n2
                    break
    
    elif mode == "reflect":
        # Reflection symmetry across vertical center
        max_c = max(Gd.nodes[n]["coords"][1] for n in Gd.nodes())
        for node in Gd.nodes():
            r, c = Gd.nodes[node]["coords"]
            target = (r, max_c - c)
            for n2 in Gd.nodes():
                if Gd.nodes[n2]["coords"] == target and n2 != node:
                    pairs[node] = n2
                    break
    
    return pairs


def build_psmsl_factor(
    Gd: nx.Graph,
    Gl: nx.Graph,
    config: PSMSLConfig,
    mirror_pairs: Dict,
) -> SpinEBMFactor:
    """
    Build PSMSL energy-based model factor.
    
    Args:
        Gd: Data plane graph
        Gl: Latent plane graph
        config: PSMSL configuration
        mirror_pairs: Dictionary of mirror relationships
    
    Returns:
        SpinEBMFactor encoding PSMSL energy function
    """
    data_nodes = list(Gd.nodes())
    latent_nodes = list(Gl.nodes())
    spin_nodes = data_nodes + latent_nodes
    index = {n: i for i, n in enumerate(spin_nodes)}
    
    pair_edges = []
    weights = []
    
    # Data plane coherence (local spatial coupling)
    for (u, v) in Gd.edges():
        pair_edges.append((index[u], index[v]))
        weights.append(config.j_local)
    
    # Latent plane coherence
    for (u, v) in Gl.edges():
        pair_edges.append((index[u], index[v]))
        weights.append(config.j_latent)
    
    # Dyad ties (data-latent vertical coupling)
    for dn in data_nodes:
        r, c = Gd.nodes[dn]["coords"]
        ln = next(n for n in latent_nodes if Gl.nodes[n]["coords"] == (r, c))
        pair_edges.append((index[dn], index[ln]))
        weights.append(config.j_dyad)
    
    # Mirror ties (geometric constraint)
    for dn in data_nodes:
        if dn in mirror_pairs:
            mnode = mirror_pairs[dn]
            pair_edges.append((index[dn], index[mnode]))
            weights.append(config.j_mirror)
    
    # Biases
    h = jnp.zeros(len(spin_nodes), dtype=jnp.float32)
    data_indices = jnp.array([index[dn] for dn in data_nodes], dtype=jnp.int32)
    latent_indices = jnp.array([index[ln] for ln in latent_nodes], dtype=jnp.int32)
    h = h.at[data_indices].set(config.h_data)
    h = h.at[latent_indices].set(config.h_latent)
    
    # Create node blocks for pairwise interactions
    node_groups_u = []
    node_groups_v = []
    for (u_idx, v_idx) in pair_edges:
        node_groups_u.append(spin_nodes[u_idx])
        node_groups_v.append(spin_nodes[v_idx])
    
    # Build factors: separate bias factors for data and latent
    data_bias_factor = SpinEBMFactor([Block(data_nodes)], h[data_indices])
    latent_bias_factor = SpinEBMFactor([Block(latent_nodes)], h[latent_indices])
    
    if len(node_groups_u) > 0:
        pair_factor = SpinEBMFactor(
            [Block(node_groups_u), Block(node_groups_v)],
            jnp.array(weights, dtype=jnp.float32)
        )
        return [data_bias_factor, latent_bias_factor, pair_factor]
    else:
        return [data_bias_factor, latent_bias_factor]


def build_psmsl_model(
    config: PSMSLConfig,
    mirror_mode: str = "phi",
) -> Tuple[SpinEBMFactor, List[Block], Dict]:
    """
    Build complete PSMSL model with blocks for sampling.
    
    Args:
        config: PSMSL configuration
        mirror_mode: Type of mirror mapping ("simple", "phi", "reflect")
    
    Returns:
        Tuple of (factor, free_blocks, node_shape_dtypes)
    """
    # Create data and latent grids
    Gd = relabel_grid_to_nodes(make_grid(config.rows, config.cols), DataSpin)
    Gl = relabel_grid_to_nodes(make_grid(config.rows, config.cols), LatentSpin)
    
    # Build mirror pairs
    mirror_pairs = build_phi_mirror_pairs(Gd, mode=mirror_mode)
    
    # Build energy factors (returns list of factors)
    factors = build_psmsl_factor(Gd, Gl, config, mirror_pairs)
    
    # Create bipartite blocks for parallel sampling
    d0, d1 = bipartite_blocks(Gd)
    l0, l1 = bipartite_blocks(Gl)
    free_blocks = [Block(d0), Block(d1), Block(l0), Block(l1)]
    
    # Node type specifications
    node_shape_dtypes = {
        DataSpin: jnp.int8,
        LatentSpin: jnp.int8,
    }
    
    return factors, free_blocks, node_shape_dtypes


class PSMSLDenoiser:
    """Multi-layer denoising model using PSMSL structure."""
    
    def __init__(
        self,
        config: PSMSLConfig,
        layers: int = 3,
        steps_per_layer: int = 250,
        mirror_mode: str = "phi",
    ):
        """
        Initialize PSMSL denoiser.
        
        Args:
            config: PSMSL configuration
            layers: Number of denoising layers
            steps_per_layer: Gibbs sampling steps per layer
            mirror_mode: Type of mirror mapping
        """
        self.config = config
        self.layers = layers
        self.steps_per_layer = steps_per_layer
        self.mirror_mode = mirror_mode
        
        # Build model components
        self.factors, self.free_blocks, self.node_shape_dtypes = build_psmsl_model(
            config, mirror_mode
        )
    
    def get_sampling_program(self):
        """Get the sampling program for this denoiser."""
        from ..block_sampling import BlockSamplingProgram
        
        conditionals = {
            DataSpin: SpinGibbsConditional(),
            LatentSpin: SpinGibbsConditional(),
        }
        
        program = BlockSamplingProgram(
            free_blocks=self.free_blocks,
            clamped_blocks=[],
            node_shape_dtypes=self.node_shape_dtypes,
            factors=self.factors,
            conditionals=conditionals,
        )
        
        return program
