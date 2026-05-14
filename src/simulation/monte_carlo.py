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

class MonteCarloChannel:
    """
    Particle-based Monte Carlo (MUCIN-like) Molecular Communication Channel.
    Simulates individual molecule random walks (Brownian motion) in 3D space.
    """
    def __init__(self, params: dict = None, seed: int = 42):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.D_M = p["D_M"]
        self.r0 = p["r0"]
        self.L = p["domain_size"]
        self.r_ref = p["r_ref"]
        self.dt = p["dt"]
        self.k_deg = p["k_deg"]
        self.flow_vel = p["flow_vel"]
        self.N_molecules = p["N_molecules"]
        self.rng = np.random.default_rng(seed)

    def simulate(self, bits: np.ndarray, Tb: float) -> np.ndarray:
        n_bits = len(bits)
        steps_per_bit = int(Tb / self.dt)
        total_steps = n_bits * steps_per_bit
        
        # Receiver at origin (0,0,0)
        # TX at distance r_ref on x-axis
        tx_pos = np.array([self.r_ref, 0.0, 0.0])
        
        # Track active molecules: list of arrays of positions
        active_molecules = []
        
        signal = np.zeros(n_bits)
        
        # Standard deviation for random walk in each dimension: sqrt(2 * D * dt)
        sigma = np.sqrt(2 * self.D_M * self.dt)
        # Degradation survival probability per step (Exponential decay)
        surv_prob = np.exp(-self.k_deg * self.dt)
        
        step = 0
        for b_idx in range(n_bits):
            # Emission at the start of the bit slot
            if bits[b_idx] == 1:
                new_mols = np.tile(tx_pos, (self.N_molecules, 1))
                active_molecules.append(new_mols)
            
            hits_in_bit = 0
            
            for _ in range(steps_per_bit):
                step += 1
                if not active_molecules:
                    continue
                
                # Combine all active molecules into one array for vectorized updates
                all_pos = np.vstack(active_molecules)
                
                # 1. Biological Degradation (Enzymatic breakdown)
                alive_mask = self.rng.random(len(all_pos)) < surv_prob
                all_pos = all_pos[alive_mask]
                
                if len(all_pos) == 0:
                    active_molecules = []
                    continue

                # 2. Advection / Bulk drift (e.g. Mucociliary clearance along Y-axis)
                all_pos[:, 1] += self.flow_vel * self.dt
                
                # 3. Random walk (Diffusion)
                all_pos += self.rng.normal(0, sigma, size=all_pos.shape)

                
                # Reflective boundaries at -L and L
                all_pos = np.where(all_pos > self.L, 2*self.L - all_pos, all_pos)
                all_pos = np.where(all_pos < -self.L, -2*self.L - all_pos, all_pos)
                
                # Absorption/Counting at receiver (spherical boundary)
                dist_to_rx = np.linalg.norm(all_pos, axis=1)
                absorbed = dist_to_rx <= self.r0
                
                hits_in_bit += np.sum(absorbed)
                
                # Keep only molecules that haven't been absorbed
                survivors = all_pos[~absorbed]
                
                # Re-pack into list (just one chunk now)
                if len(survivors) > 0:
                    active_molecules = [survivors]
                else:
                    active_molecules = []
            
            # Normalize signal based on total emitted per bit for ease of detection
            signal[b_idx] = hits_in_bit / self.N_molecules
            
        return signal
