# Molecular Communication — ISI Detection Benchmark

Final project for CMPE49G. Benchmarks classical and ML-based detectors for
diffusion-based mobile molecular communication (MC) channels, and introduces
a novel **Hybrid ISI-Cancel + RC** detector.

## Papers

| Paper | Description |
|-------|-------------|
| Meng et al. 2014 (IEEE TSP) | ISI cancellation for static 3D diffusion channel |
| Bilge et al. 2025 (arXiv) | Reservoir Computing detector for mobile MC |

## Architecture

```
molecular-communication/
├── src/
│   ├── simulation/
│   │   ├── channel.py           # Mobile MC channel (Brownian TX/RX motion + ISI)
│   │   └── signal_generator.py  # Bit sequence generation and train/test split
│   ├── detectors/
│   │   ├── base.py              # BaseDetector interface
│   │   ├── classical.py         # FixedThreshold, AdaptiveEMA, MAP (mismatched)
│   │   ├── ml_detectors.py      # MLP, ANN, CNN, LSTM (PyTorch)
│   │   ├── reservoir.py         # RC, RC-ISI (Echo State Network, Bilge 2025)
│   │   └── hybrid.py            # [NOVEL] ISI Cancel (Meng 2014) + RC-ISI
│   ├── preprocessing/
│   │   └── isi_cancellation.py  # Meng 2014 ISI cancellation: z[k] = u[k] - u[k-1]
│   └── evaluation/
│       └── metrics.py           # BER, accuracy, ROC AUC, latency, param count
└── experiments/
    ├── run_all.py               # Full benchmark across Tb values
    └── plot_results.py          # Generate Figs 4/5 from Bilge 2025 + hybrid comparison
```

## Detectors

| Detector | Type | Params |
|----------|------|--------|
| Peak-Fixed | Classical threshold | 0 |
| Adaptive-EMA | EMA threshold (Damrath 2016) | 0 |
| MAP (Mismatched) | Viterbi with static avg CIR | 0 |
| ANN | Shallow feedforward NN | ~5k–66k |
| CNN | 1-D conv network | 465 |
| LSTM | Recurrent network | ~5k |
| Feedforward-MLP | Deep MLP | 21k–264k |
| RC | Echo State Network (fixed 0.5 threshold) | 401 |
| RC-ISI | ESN + ROC threshold optimization | 401 |
| **Hybrid-ISICancel+RC** | ISI cancel preprocessing + RC-ISI | 401 |

## Novel Extension

The `HybridISICancelRC` detector applies the Meng 2014 ISI cancellation step
(`z[k] = u[k] − u[k−1]`) as a preprocessing filter before feeding the cleaned
signal into the RC-ISI reservoir. The hypothesis is that physics-informed
preprocessing reduces the dynamic range the reservoir must cover, improving
accuracy under severe ISI (low Tb) without increasing model complexity.

## Quick Start

```bash
pip install -r requirements.txt

# Fast smoke test (500 bits, selected Tb values)
python experiments/run_all.py --fast --tb 10 100 200

# Full benchmark (2000 bits, all Tb values from paper)
python experiments/run_all.py

# Generate figures
python experiments/plot_results.py
```

## Channel Model

The channel captures:
- **ISI via exponential decay**: memory parameter ρ = exp(−Tb/τ) where τ = L²/D_M
- **Mobility**: TX–RX distance performs a Brownian random walk at symbol rate
- **Noise**: additive Gaussian noise (std=0.03)

Higher Tb → smaller ρ → less ISI → easier detection.
Lower Tb → larger ρ → more ISI → detectors approach random guessing.

## Expected Results (qualitative)

At Tb=10s (severe ISI): all detectors ~55–70% accuracy.
At Tb=100s (moderate ISI): RC-ISI ~87%, Hybrid ~97%, classical ~93–96%.
At Tb=200s (low ISI): most learned models converge to >95%.

## Use Cases

- **Smart Transmitter, Dumb Receiver**: In this scenario, complex processing (e.g., machine learning models, pre-coding, or ISI cancellation) is performed at the transmitter. The receiver uses a simple, low-complexity detector (like `Peak-Fixed` or `Adaptive-EMA`) to decode the signal. This architecture is ideal for nanonetworks where the receiver (e.g., in-body nanoscale devices) has extreme energy, hardware, and computational limitations, while the transmitter (e.g., external gateway) has access to abundant resources.
