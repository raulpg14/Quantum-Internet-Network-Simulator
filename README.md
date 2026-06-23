# Quantum Communication Networks — OFBQI & SBQI Simulation

![CI](https://github.com/raulpg14/Quantum-Internet-Network-Simulator/actions/workflows/ci.yml/badge.svg)
[![Open In Colab — Degree Distribution](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raulpg14/Quantum-Internet-Network-Simulator/blob/master/notebooks/01_degree_distribution.ipynb)
[![Open In Colab — Evolution Analysis](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raulpg14/Quantum-Internet-Network-Simulator/blob/master/notebooks/02_evolution_analysis.ipynb)

A scientific simulation tool for analysing the structural properties of **OFBQI** (Optical Fiber-Based Quantum Internet) and **SBQI** (Satellite-Based Quantum Internet) network architectures. The tool generates random quantum network topologies using physics-based probabilistic connection models and analyses their degree distributions, average path lengths, and diameters at scale.

---

## Quick start — local

```bash
git clone https://github.com/raulpg14/Quantum-Internet-Network-Simulator.git
cd Quantum-Internet-Network-Simulator
pip install -e ".[dev]"
```

Launch the desktop GUI:
```bash
python -m qcn.interfaces.gui.main
```

Open the analysis notebooks:
```bash
jupyter lab notebooks/
```

---

## Quick start — Google Colab

No local installation required. Click a badge above to open a notebook directly in Colab. The notebook installs the package automatically on first run.

---

## Network types

| Type | Connection model | Degree distribution |
|------|-----------------|-------------------|
| OFBQI | Two-stage: optical fiber attenuation → photonic filter | Poisson |
| SBQI | Satellite free-space optical link probability | Log-normal |

---

## Simulation modes

**Degree Distribution** — generates a network at fixed radius and analyses the degree distribution across Monte Carlo repetitions.

**Evolution (N)** — sweeps the number of nodes N at fixed density, tracking how average shortest path length and diameter scale with network size.

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Project structure
qcn/

engine/        # simulation core — Network, Node, physics functions, config

interfaces/

gui/         # local Tkinter desktop application

notebooks/       # Colab-ready Jupyter notebooks

tests/           # pytest test suite

data/            # simulation input/output data (gitignored)

results/         # simulation results (gitignored)

---

## Old version

The original version of this project is preserved under the `old-version` git tag.