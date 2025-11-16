# THRML + PSMSL: Geometric Energy-Based Models

This is an enhanced fork of [THRML](https://github.com/extropic-ai/thrml) that integrates **PSMSL (Projected Symmetry Mirrored Semantic Lattice)** geometric computation concepts into the thermodynamic computing framework.

## What is PSMSL-THRML?

PSMSL-THRML provides a framework for encoding geometric constraints—mirror symmetry, phi-scaling, and spatial structure—directly into energy-based models that can run on thermodynamic sampling units (TSUs).

### Key Features

- **Geometric Priors**: Encode mirror symmetry and golden ratio relationships as energy function constraints
- **Dual-Plane Architecture**: Colocated data and latent planes with vertical coupling
- **Multiple Mirror Modes**: Simple pooling, phi-scaling, or reflection symmetry
- **Block Gibbs Sampling**: Parallel updates via bipartite graph coloring
- **Multi-Layer Denoising**: Progressive refinement through sampling layers
- **TSU-Compatible**: Runs on GPU simulation today, TSU hardware tomorrow

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/psmsl-thrml-enhanced.git
cd psmsl-thrml-enhanced
pip install -e .
```

## Quick Start

### Basic PSMSL Model

```python
from thrml.models import PSMSLConfig, build_psmsl_model, SpinGibbsConditional
from thrml.models.psmsl import DataSpin, LatentSpin
from thrml.block_sampling import BlockSamplingProgram, sample_states
import jax

# Configure model
config = PSMSLConfig(
    rows=16,
    cols=16,
    j_local=0.6,    # Data-to-data coupling
    j_latent=0.4,   # Latent-to-latent coupling
    j_dyad=0.8,     # Data-to-latent coupling
    j_mirror=0.25,  # Mirror symmetry coupling
)

# Build model with phi-scaling
factor, free_blocks, node_shape_dtypes = build_psmsl_model(
    config, mirror_mode="phi"
)

# Create sampling program
conditionals = {
    DataSpin: SpinGibbsConditional(),
    LatentSpin: SpinGibbsConditional(),
}

program = BlockSamplingProgram(
    free_blocks=free_blocks,
    clamped_blocks=[],
    node_shape_dtypes=node_shape_dtypes,
    factors=[factor],
    conditionals=conditionals,
)

# Sample
final_states, _ = sample_states(
    program=program,
    init_state=init_states,  # Initialize as shown in examples
    schedule=program.default_schedule(),
    n_steps=500,
    keys=jax.random.split(jax.random.key(42), n_chains),
)
```

### Multi-Layer Denoising

```python
from thrml.models import PSMSLDenoiser

# Create denoiser
denoiser = PSMSLDenoiser(
    config=config,
    layers=3,
    steps_per_layer=200,
    mirror_mode="phi",
)

# Get sampling program and run
program = denoiser.get_sampling_program()
# ... run sampling as above
```

## Examples

Run the demonstration scripts:

```bash
# Simple Python demo
python examples/demo_psmsl.py

# Jupyter notebook with visualizations
jupyter notebook examples/03_psmsl_geometric_ebm.ipynb
```

## Architecture

### Energy Function

The PSMSL energy function combines multiple coupling types:

```
E(s) = -Σ j_local·s_i·s_j          (data plane coherence)
       -Σ j_latent·s_k·s_l         (latent plane coherence)
       -Σ j_dyad·s_i·s_k           (data-latent coupling)
       -Σ j_mirror·s_i·s_m(i)      (mirror constraints)
       -Σ h_data·s_i - Σ h_latent·s_k
```

Where:
- `s_i, s_j` are data plane spins
- `s_k, s_l` are latent plane spins
- `m(i)` is the mirror partner of spin `i`

### Mirror Modes

**Simple**: Column pooling (placeholder)
```python
mirror_mode="simple"  # Maps (r,c) → (r, c//2)
```

**Phi-Scaling**: Golden ratio indexing
```python
mirror_mode="phi"  # Maps (r,c) → (r, c/φ) where φ=1.618...
```

**Reflection**: Symmetric mirroring
```python
mirror_mode="reflect"  # Maps (r,c) → (r, max_c-c)
```

## Applications

PSMSL-THRML is designed for problems with inherent geometric structure:

### Physics Simulation
- Magnetic systems with symmetry (Ising models)
- Crystal structure prediction
- Molecular conformation sampling
- Materials science

### Generative Modeling
- Geometric pattern synthesis
- Texture generation with constraints
- Architectural design
- Artistic pattern creation

### Constrained Optimization
- Layout design with symmetry requirements
- Resource allocation with spatial constraints
- Circuit design with geometric rules

### Denoising and Reconstruction
- Image denoising preserving symmetry
- Signal reconstruction maintaining structure
- Data imputation with geometric consistency

## Comparison to Standard THRML

| Feature | Standard THRML | PSMSL-THRML |
|---------|---------------|-------------|
| **Graph Structure** | Generic | Dual-plane with mirrors |
| **Symmetry** | Manual | Built-in (configurable) |
| **Latent Variables** | Optional | Colocated with data |
| **Scaling** | Arbitrary | Phi-based indexing |
| **Use Cases** | General EBMs | Geometry-aware problems |

## Performance

PSMSL models inherit THRML's performance characteristics:

- **GPU Simulation**: Runs efficiently on JAX/GPU today
- **TSU Hardware**: Compatible with future thermodynamic sampling units
- **Energy Efficiency**: Potential 10,000x improvement on TSU vs GPU
- **Parallel Sampling**: Block Gibbs enables parallel updates

## Contributing

This is a demonstration fork showcasing PSMSL integration. Contributions welcome:

1. Enhanced mirror mapping algorithms
2. Additional geometric constraints
3. Domain-specific applications
4. Performance optimizations
5. Documentation improvements

## Citation

If you use PSMSL-THRML in your research, please cite both THRML and this work:

```bibtex
@misc{jelinčič2025efficientprobabilistichardwarearchitecture,
      title={An efficient probabilistic hardware architecture for diffusion-like models}, 
      author={Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and Guillaume Verdon and Trevor McCourt},
      year={2025},
      eprint={2510.23972},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23972}, 
}

@software{psmsl_thrml_2025,
  title={PSMSL-THRML: Geometric Energy-Based Models for Thermodynamic Computing},
  author={[Your Name]},
  year={2025},
  url={https://github.com/YOUR_USERNAME/psmsl-thrml-enhanced}
}
```

## License

Apache 2.0 (same as THRML)

## Acknowledgments

- **Extropic AI** for the THRML framework and thermodynamic computing research
- **PSMSL** for geometric computation concepts
- The broader thermodynamic and probabilistic computing community

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [Link to your repo]
- Email: [Your email]
- Twitter/X: [Your handle]

---

**Status**: Demonstration/Research Code

This fork demonstrates the integration of geometric constraints into thermodynamic computing. It is intended for research, experimentation, and showcasing the potential of PSMSL concepts on TSU-compatible platforms.
