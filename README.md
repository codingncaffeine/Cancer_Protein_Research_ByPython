# 🧬 Cancer Protein Research Folding @ Home Application 

### Real Molecular Dynamics for Cancer Protein Research

This application performs **real molecular dynamics (MD)** simulations of cancer-related proteins using [OpenMM](https://openmm.org) and visualizes results interactively.  
It provides an intuitive interface for both researchers and students to explore **protein motion, stability, and drug interaction potential** through modern GPU-accelerated simulations.

---

## 🚀 Features

- **Real MD simulations** using the AMBER14 force field  
- **Interactive visualization** of protein folding and movement  
- **Cancer protein library** with curated PDB IDs (p53, BRCA1, EGFR, RAS, HER2, etc.)  
- **Live metrics:** RMSD, temperature, and potential energy  
- **Automatic export:** PDB structure, trajectory data, statistical summary, and README  
- **GPU acceleration** via CUDA or OpenCL (auto-detected)  
- **Cross-platform:** Windows, macOS, Linux  

---

## 🧩 Why This Research Matters

Protein dynamics — not just static structures — determine **how cancer mutations disrupt biological function** and **how drugs bind or fail**.  
By contributing your simulation data, you help:

- Build a **global dataset** of cancer protein motions  
- Enable **AI and statistical models** for drug-binding prediction  
- Improve **reproducibility** and open access to molecular data  
- Support **comparative studies** between healthy and mutant proteins  

This project contributes to open cancer research by transforming *raw MD data* into *shared, reusable knowledge*.

---

## 🖥️ Installation

### Requirements
Install dependencies via **conda** and **pip**:

```bash
conda install -c conda-forge openmm pdbfixer
pip install biopython requests numpy
