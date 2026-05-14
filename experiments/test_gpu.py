import sys
import os
import time
import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.simulation.monte_carlo_gpu import MonteCarloChannelGPU
from src.simulation.signal_generator import generate_bits

def test_gpu_simulation():
    print("Testing GPU Monte Carlo Simulation...")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    n_bits = 100
    Tb = 10.0
    bits = generate_bits(n_bits, seed=42)
    
    # Use larger number of molecules to showcase GPU capability
    params = {"N_molecules": 5000}
    
    print(f"Initializing Channel with {params['N_molecules']} molecules per bit...")
    channel = MonteCarloChannelGPU(params=params, seed=42)
    print(f"Using device: {channel.device}")
    
    start_time = time.time()
    signal = channel.simulate(bits, Tb)
    end_time = time.time()
    
    print("\n--- Simulation Results ---")
    print(f"Total time   : {end_time - start_time:.2f} seconds")
    print(f"Bits sent    : {n_bits}")
    print(f"Signal shape : {signal.shape}")
    print(f"Sample signal: {signal[:10]}")

if __name__ == "__main__":
    test_gpu_simulation()
