# PyMC Gravitational Wave Binary System Inference

This repository provides a framework for hierarchical Bayesian inference of binary system parameters (m1, m2, six spin components, chirp mass, chi effective) using PyMC.

**Use Case:** Start with mock data to validate model structure, then integrate with PSO or real GW data.

**Parameters Inferred:** Masses (m1, m2), spins (s1x, s1y, s1z, s2x, s2y, s2z), chirp mass, chi effective.

## Getting Started

1. Install requirements: `pip install -r requirements.txt`
2. Generate mock data: `python scripts/generate_mock_data.py`
3. Run inference: `python scripts/run_inference.py`
4. Explore results in `notebooks/demo_inference.ipynb`