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

## 🗂️ Repository Structure

## 📝 Citation

If you use this code or datasets in your research, please cite the following paper:

Nabipour, I., Raoof, A., & Qajar, J. (2026). Resolution-aware multiscale 3D CNNs for permeability estimation in carbonate rocks: an architecture-agnostic transfer-learning framework. *Advances in Water Resources*, 182, 105311.  
https://doi.org/10.1016/j.advwatres.2026.105311

**BibTeX:**
```bibtex
@article{Nabipour2026ResolutionAware,
  title={Resolution-aware multiscale 3D CNNs for permeability estimation in carbonate rocks: an architecture-agnostic transfer-learning framework},
  author={Nabipour, Iman and Raoof, Amir and Qajar, Jafar},
  journal={Advances in Water Resources},
  volume={182},
  pages={105311},
  year={2026},
  doi={10.1016/j.advwatres.2026.105311}
}



## 📜 License

This project is released under the MIT License.  
See the `LICENSE` file for details.

## 📬 Contact

For questions, discussions, or collaborations, feel free to contact:

**Iman Nabipour**  
Email: i.nabipour1988@gmail.com  
GitHub: https://github.com/ImanNabipour


Download Dataset:

Google drive link for the 3D normalized distance maps carbonate micro-CT images: https://drive.google.com/file/d/1G2HMvpO0z4J9GwioNVdElAW6B-nTVY5B/view?usp=sharing

Google drive link for the 3D binary maps carbonate micro-CT images: https://drive.google.com/file/d/1brAfMJFHogrkf842fn_DUb2lYNXV-cux/view?usp=sharing

After downloading, extract the dataset and place it in a data/ directory.

