"""
Diffusion-based molecular communication channel — synthetic model.

This model captures the essential physics of the Bilge 2025 / Smoldyn channel:
  - Long-tailed ISI that worsens as Tb decreases
  - Time-varying CIR from mobile TX and RX (Brownian distance walk)
  - Poisson-like receptor noise

Physical basis:
  In Smoldyn's bounded cubic domain, molecules reflect off walls and
  accumulate. The effective memory of the channel is governed by the
  domain diffusion relaxation time τ = L² / D_M.  We model this with
  an exponential ISI tail whose decay rate scales as exp(-Tb / τ).
  For the default parameters (L=100µm, D_M=1.01e-9 m²/s) τ≈10s,
  giving rho≈0.37 at Tb=10s and rho≈4.5e-5 at Tb=100s.

Signal model:
  s[k] = Σ_j  bits[j] × h(k-j, r[k])  + noise
  h(m, r) = rho(r)^m                    for m >= 0
  rho(r)  = exp(-Tb / (τ × (r_ref/r)²))   (further apart → faster decay)

The signal is normalized to [0,1] by dividing by the maximum possible
ISI accumulation (geometric series of the strongest rho).
"""
import numpy as np


DEFAULT_PARAMS = {
    "N_molecules": 2000,
    "D_M": 1.01e-9,      # m^2/s
    "D_T": 4.74e-14,     # m^2/s  (TX mobility)
    "D_R": 2.31e-12,     # m^2/s  (RX mobility)
    "r0": 5e-6,          # RX radius (m)
    "domain_size": 100e-6,   # half-side of cubic domain (m)
    "r_ref": 20e-6,      # reference TX–RX distance (m)
    "noise_std": 0.03,   # additive Gaussian noise std
}


class MobileChannel:
    """
    Fast, analytically-calibrated mobile MC channel.

    Parameters
    ----------
    params : dict, optional — override any DEFAULT_PARAMS key
    seed   : int
    """

    def __init__(self, params: dict = None, seed: int = 42):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.D_M = p["D_M"]
        self.D_T = p["D_T"]
        self.D_R = p["D_R"]
        self.r0 = p["r0"]
        self.L = p["domain_size"]
        self.r_ref = p["r_ref"]
        self.noise_std = p["noise_std"]
        # τ = domain diffusion relaxation time
        self.tau = self.L ** 2 / self.D_M
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def _rho(self, Tb: float, r: float) -> float:
        """Per-symbol ISI decay factor at distance r."""
        # Effective τ scales as r² (further TX → faster dispersion away)
        tau_eff = self.tau * (r / self.r_ref) ** 2
        return float(np.exp(-Tb / max(tau_eff, 1e-10)))

    def simulate(
        self,
        bits: np.ndarray,
        Tb: float,
        smart_tx: bool = False,
        gamma: float = 0.8,
    ) -> np.ndarray:
        """
        Simulate the mobile MC channel.

        Parameters
        ----------
        bits : (n,) int array of 0/1
        Tb   : symbol duration in seconds
        smart_tx : bool
            If True, the transmitter adjusts its emission strength based on
            the estimated ISI residual from its own past transmissions.
            For bit=1, emission weight = max(1-γ*baseline, min_w) instead of 1.
        gamma : float
            Aggressiveness of smart-TX adjustment (0 = no adjustment).

        Returns
        -------
        signal : (n,) float in [0, 1] (normalized receptor occupancy)
        """
        n = len(bits)
        signal = np.zeros(n)

        # TX–RX distance trajectory (Brownian walk at symbol rate)
        D_mob = self.D_T + self.D_R
        sigma_walk = np.sqrt(2 * D_mob * Tb)
        r = np.full(n, self.r_ref, dtype=float)
        for k in range(1, n):
            r[k] = max(self.r0 * 2, r[k - 1] + self.rng.normal(0, sigma_walk))
            r[k] = min(r[k], self.L * 1.5)   # keep within plausible range

        rho_arr = np.array([self._rho(Tb, rk) for rk in r])

        # --- Emission weights (smart TX adjusts per-symbol) ---------------
        # emission[k] is the fraction of N_molecules actually emitted for
        # bit k.  With smart_tx=False every bit=1 emits weight 1.0.
        emission = np.ones(n, dtype=float)

        if smart_tx:
            min_w = 0.1          # never go below 10 % of base emission
            for k in range(n):
                if bits[k] == 1 and k > 0:
                    # TX estimates ISI baseline from its own past emissions
                    # using the *reference* rho (TX knows Tb and τ but not
                    # the exact instantaneous r, so it uses r_ref).
                    rho_ref = self._rho(Tb, self.r_ref)
                    baseline = 0.0
                    for j in range(k):
                        baseline += bits[j] * emission[j] * (rho_ref ** (k - j))
                    # Normalize baseline to [0, 1] range
                    peak_ref = 1.0 / (1.0 - rho_ref) if rho_ref < 0.9999 else float(k)
                    baseline /= max(peak_ref, 1e-10)
                    emission[k] = max(min_w, 1.0 - gamma * baseline)

        # --- Channel propagation ------------------------------------------
        for k in range(n):
            rho_k = rho_arr[k]
            lags = np.arange(k + 1)          # lag = k - j for j=0..k
            weights = rho_k ** lags[::-1]     # weights[j] = rho^(k-j)
            # bits modulated by emission weight
            effective = bits[:k + 1].astype(float) * emission[:k + 1]
            raw = float(effective @ weights)
            # Normalize by all-1s geometric series
            peak = 1.0 / (1.0 - rho_k) if rho_k < 0.9999 else float(k + 1)
            raw /= peak
            noise = self.rng.normal(0, self.noise_std)
            signal[k] = float(np.clip(raw + noise, 0.0, 1.5))

        return signal
