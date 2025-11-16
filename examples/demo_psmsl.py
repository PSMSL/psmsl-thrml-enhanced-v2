"""
Simple demonstration of PSMSL geometric energy-based models on THRML.
"""

import jax
import jax.numpy as jnp
import numpy as np

from thrml.models import (
    PSMSLConfig,
    PSMSLDenoiser,
    build_psmsl_model,
    SpinGibbsConditional,
)
from thrml.models.psmsl import DataSpin, LatentSpin
from thrml.block_sampling import BlockSamplingProgram, sample_states


def demo_single_layer():
    """Demonstrate single-layer PSMSL sampling."""
    print("=" * 60)
    print("PSMSL Single-Layer Demo")
    print("=" * 60)
    
    # Configure model
    config = PSMSLConfig(
        rows=8,
        cols=8,
        j_local=0.6,
        j_latent=0.4,
        j_dyad=0.8,
        j_mirror=0.25,
    )
    
    print(f"\nBuilding {config.rows}x{config.cols} PSMSL model...")
    
    # Build model with phi-scaling
    factor, free_blocks, node_shape_dtypes = build_psmsl_model(
        config, mirror_mode="phi"
    )
    
    print(f"  Nodes: {len(factor.nodes)}")
    print(f"  Pairwise couplings: {len(factor.pair_indices)}")
    print(f"  Free blocks: {len(free_blocks)}")
    
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
    
    # Initialize states
    n_chains = 16
    n_steps = 500
    rng = jax.random.key(42)
    
    print(f"\nInitializing {n_chains} chains...")
    init_states = []
    keys = jax.random.split(rng, len(free_blocks) * n_chains + 1)
    ki = 0
    
    for block in free_blocks:
        block_shape = (n_chains, len(block.nodes))
        spins = jax.random.choice(
            keys[ki], jnp.array([-1, 1], dtype=jnp.int8), shape=block_shape
        )
        init_states.append(spins)
        ki += 1
    
    # Run sampling
    print(f"Running {n_steps} Gibbs sampling steps...")
    sample_keys = jax.random.split(rng, n_chains)
    
    final_states, _ = sample_states(
        program=program,
        init_state=init_states,
        schedule=program.default_schedule(),
        n_steps=n_steps,
        keys=sample_keys,
        observers=[],
        return_observations=True,
    )
    
    # Analyze results
    print("\nResults:")
    for i, state in enumerate(final_states):
        mean_spin = np.array(state).mean()
        std_spin = np.array(state).std()
        print(f"  Block {i}: mean={mean_spin:.3f}, std={std_spin:.3f}")
    
    print("\n✓ Single-layer demo completed successfully")
    return final_states


def demo_denoising():
    """Demonstrate multi-layer denoising."""
    print("\n" + "=" * 60)
    print("PSMSL Multi-Layer Denoising Demo")
    print("=" * 60)
    
    # Configure denoiser
    config = PSMSLConfig(rows=8, cols=8)
    denoiser = PSMSLDenoiser(
        config=config,
        layers=3,
        steps_per_layer=200,
        mirror_mode="phi",
    )
    
    print(f"\nCreated denoiser with {denoiser.layers} layers")
    print(f"  Steps per layer: {denoiser.steps_per_layer}")
    print(f"  Mirror mode: {denoiser.mirror_mode}")
    
    # Get sampling program
    program = denoiser.get_sampling_program()
    
    # Initialize
    n_chains = 16
    rng = jax.random.key(42)
    
    init_states = []
    keys = jax.random.split(rng, len(denoiser.free_blocks) * n_chains + 1)
    ki = 0
    
    for block in denoiser.free_blocks:
        block_shape = (n_chains, len(block.nodes))
        spins = jax.random.choice(
            keys[ki], jnp.array([-1, 1], dtype=jnp.int8), shape=block_shape
        )
        init_states.append(spins)
        ki += 1
    
    # Run denoising layers
    print("\nRunning denoising layers...")
    state = init_states
    sample_keys = jax.random.split(rng, n_chains)
    trajectory = []
    
    for layer in range(denoiser.layers):
        state, _ = sample_states(
            program=program,
            init_state=state,
            schedule=program.default_schedule(),
            n_steps=denoiser.steps_per_layer,
            keys=sample_keys,
            observers=[],
            return_observations=True,
        )
        trajectory.append(state)
        
        # Analyze this layer
        mean_spin = np.array(state[0]).mean()
        print(f"  Layer {layer + 1}: data plane mean = {mean_spin:.3f}")
    
    print("\n✓ Denoising demo completed successfully")
    return trajectory


def demo_mirror_modes():
    """Compare different mirror modes."""
    print("\n" + "=" * 60)
    print("PSMSL Mirror Mode Comparison")
    print("=" * 60)
    
    config = PSMSLConfig(rows=8, cols=8)
    modes = ["simple", "phi", "reflect"]
    
    for mode in modes:
        print(f"\nTesting mirror mode: {mode}")
        
        factor, free_blocks, _ = build_psmsl_model(config, mirror_mode=mode)
        
        # Count mirror couplings (those with j_mirror weight)
        mirror_count = sum(
            1 for w in factor.pair_weights if abs(w - config.j_mirror) < 1e-6
        )
        
        print(f"  Total couplings: {len(factor.pair_indices)}")
        print(f"  Mirror couplings: {mirror_count}")
    
    print("\n✓ Mirror mode comparison completed")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PSMSL-THRML Integration Demonstration")
    print("=" * 60)
    print("\nThis demonstrates geometric energy-based models")
    print("integrated into the THRML thermodynamic computing framework.")
    print()
    
    # Run demos
    demo_single_layer()
    demo_denoising()
    demo_mirror_modes()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
    print("\nPSMSL models are now integrated with THRML and ready for:")
    print("  • Geometric pattern generation")
    print("  • Physics simulation with symmetry constraints")
    print("  • Constrained optimization")
    print("  • Denoising with geometric priors")
    print("\nCompatible with future TSU hardware for 10,000x energy efficiency!")
    print()


if __name__ == "__main__":
    main()
