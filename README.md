# MRAN
# Multiscale 3D CNNs for Predicting Permeability from 3D Micro-CT Carbonate Rock Images

**MRAN (Multiscale Resolution-Aware Network)** is a lightweight, custom-designed 3D convolutional neural network implemented in **PyTorch**. It leverages hierarchical transfer learning to predict permeability from multiscale, 3D micro-CT images of carbonate rocks, effectively bridging the trade-off between resolution and field-of-view. This repository includes the MRAN architecture, benchmark CNN models (3D ResNet50, ResNeXt50, DenseNet201), scripts for dataset construction, training, and evaluation, and sample datasets, enabling reproducibility and further development in digital rock physics and porous media applications.


Official implementation of the paper **"Resolution-aware multiscale 3D CNNs for permeability estimation in carbonate rocks: an architecture-agnostic transfer-learning framework"** published in *Advances in Water Resources* (2026).

## 📄 Paper Information

**Authors:** Iman Nabipour, Amir Raoof, Jafar Qajar  
**Journal:** Advances in Water Resources  
**Year:** 2026  
**DOI:** [10.1016/j.advwatres.2026.105311](https://doi.org/10.1016/j.advwatres.2026.105311)

## 🎯 Overview

This repository contains the implementation of a compact **Multiscale Resolution-Aware CNN (MRAN)** for predicting permeability from 3D micro-CT images of tight carbonate rocks. We systematically benchmark MRAN against three established 3D CNN architectures:

- **ResNet50**
- **ResNeXt50**
- **DenseNet201**

All networks are trained using a **Fine-Intermediate-Coarse (FR-IR-CR) hierarchical transfer-learning scheme** on multiresolution micro-CT images with lattice Boltzmann method (LBM)-derived permeabilities. The framework enables controlled assessment of:

- Architecture choice
- Model complexity
- Data scaling effects
