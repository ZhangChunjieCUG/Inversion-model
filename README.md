# Inversion-model

**Hybrid Probabilistic Inversion for Mixed Discrete–Continuous Subsurface Parameters**  

---
## Overview

This repository provides an open-source implementation of a **hybrid probabilistic framework** for subsurface parameter estimation.  
Our targets is the inversion of **mixed discrete–continuous parameters** (e.g., lithology, fluid type, porosity, scaling factors) from multi-physics observations, using joint seismic (Vp, Vs) and electrical conductivity (σ) data. 

**Note:** The true forward model used in the paper is not included here due to confidentiality.  
This repository only demonstrates the inversion workflow. You can apply this framework to your own research, but you will need to provide your own forward model.


**Zhang & Iwamori (2025)**  
*Hybrid Probabilistic Framework for Mixed Discrete–Continuous Subsurface Parameter Estimation from Multi-Physics Geophysical Data: A Case Study for Quantifying the Surface Geofluid Mapping*


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
git clone https://github.com/ZhangChunjieCUG/Inversion-model.git
cd Inversion-model
pip install -r requirements.txt
