import torch
import numpy as np

DEFAULT_PARAMS = {
    "N_molecules": 2000,
    "D_M": 1.01e-10,     # m^2/s (Mucin environment, ~10x slower diffusion)
    "r0": 5e-6,          # RX radius (m)
    "domain_size": 100e-6,   # half-side of cubic domain (m)
    "r_ref": 20e-6,      # reference TX-RX distance (m)
    "dt": 1e-3,          # Simulation time step (s)
    "k_deg": 0.1,        # 1/s (Enzymatic degradation rate, half-life ~ 6.9s)
    "flow_vel": 1e-6,    # m/s (Mucociliary clearance background flow)
}

class MonteCarloChannelGPU:
    """
    Particle-based Monte Carlo (MUCIN-like) Molecular Communication Channel using PyTorch (CUDA).
    Simulates individual molecule random walks (Brownian motion) in 3D space, heavily accelerated.
    """
    def __init__(self, params: dict = None, seed: int = 42, device=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.D_M = p["D_M"]
        self.r0 = p["r0"]
        self.L = p["domain_size"]
        self.r_ref = p["r_ref"]
        self.dt = p["dt"]
        self.k_deg = p["k_deg"]
        self.flow_vel = p["flow_vel"]
        self.N_molecules = p["N_molecules"]
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

    def simulate(self, bits: np.ndarray, Tb: float) -> np.ndarray:
        n_bits = len(bits)
        steps_per_bit = int(Tb / self.dt)
        
        tx_pos = torch.tensor([self.r_ref, 0.0, 0.0], device=self.device)
        
        # Track active molecules: one single tensor
        all_pos = torch.empty((0, 3), device=self.device)
        
        signal = np.zeros(n_bits)
        
        sigma = np.sqrt(2 * self.D_M * self.dt)
        surv_prob = np.exp(-self.k_deg * self.dt)
        
        for b_idx in range(n_bits):
            if bits[b_idx] == 1:
                new_mols = tx_pos.repeat(self.N_molecules, 1)
                all_pos = torch.cat([all_pos, new_mols], dim=0)
            
            hits_in_bit = 0
            
            for _ in range(steps_per_bit):
                if all_pos.shape[0] == 0:
                    continue
                
                # 1. Biological Degradation
                alive_mask = torch.rand(all_pos.shape[0], generator=self.rng, device=self.device) < surv_prob
                all_pos = all_pos[alive_mask]
                
                if all_pos.shape[0] == 0:
                    continue

                # 2. Advection / Bulk drift (Y-axis)
                all_pos[:, 1] += self.flow_vel * self.dt
                
                # 3. Random walk (Diffusion)
                all_pos += torch.randn_like(all_pos, generator=self.rng) * sigma
                
                # Reflective boundaries at -L and L
                mask_high = all_pos > self.L
                mask_low = all_pos < -self.L
                all_pos[mask_high] = 2 * self.L - all_pos[mask_high]
                all_pos[mask_low] = -2 * self.L - all_pos[mask_low]
                
                # Absorption/Counting at receiver
                dist_to_rx = torch.norm(all_pos, dim=1)
                absorbed = dist_to_rx <= self.r0
                
                hits_in_bit += int(absorbed.sum().item())
                
                # Keep survivors
                all_pos = all_pos[~absorbed]
            
            signal[b_idx] = hits_in_bit / self.N_molecules
            
        return signal
