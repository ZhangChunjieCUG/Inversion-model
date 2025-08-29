# Inversion-model

**Hybrid Probabilistic Inversion for Mixed Discrete–Continuous Subsurface Parameters**  
Using joint seismic (Vp, Vs) and electrical conductivity (σ) data.

---

## Overview

This repository provides an open-source implementation of a **hybrid probabilistic framework** for subsurface parameter estimation.  
It targets the inversion of **mixed discrete–continuous parameters** (e.g., lithology, fluid type, porosity, scaling factors) from multi-physics observations.

### Workflow

1. **Coarse-grid screening** over continuous parameters  
2. **Local sampling** using Monte Carlo / Particle Swarm Optimization (PSO)  
3. **Posterior estimation** via Kernel Density Estimation (KDE)  
4. **Discrete inference** using High-Probability Region (HPR) analysis  

This *continuous-first, discrete-second* strategy reduces error propagation from early misclassification and captures multi-modal posteriors.

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/Inversion-model.git
cd Inversion-model
pip install -r requirements.txt
