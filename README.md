# MRAN
# Multiscale 3D CNNs for Predicting Permeability from 3D Micro-CT Carbonate Rock Images


![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.advwatres.2026.105311-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)


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


### Key Features:
- **MRAN Architecture:** Novel multiscale 3D CNN designed for permeability prediction
- **Baseline Models:** ResNet50, ResNeXt50, and DenseNet201 implementations
- **Transfer Learning Framework:** Hierarchical FR→IR→CR pipeline
- **LBM Integration:** Permeability data computed using lattice Boltzmann method
- **3D Micro-CT Processing:** Specialized preprocessing for carbonate rock images

---

## 🗂️ Repository Structure
my-research-project/

├── README.md # Project documentation

├── requirements.txt # Python dependencies

├── LICENSE # MIT License

├── ProposedModel.py # MRAN model implementation

├── ResNet.py # ResNet50 3D implementation

├── ResNeXt.py # ResNeXt50 3D implementation

├── DenseNet.py # DenseNet201 3D implementation


## 📝 Citation

If you use this code or datasets in your research, please cite the following paper:

Nabipour, I., Raoof, A., & Qajar, J. (2026). Resolution-aware multiscale 3D CNNs for permeability estimation in carbonate rocks: an architecture-agnostic transfer-learning framework. *Advances in Water Resources*, 182, 105311.  
https://doi.org/10.1016/j.advwatres.2026.105311

**BibTeX:**

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


📊 Dataset

The dataset consists of 3D micro-CT images of carbonate rocks with corresponding permeability values computed using the lattice Boltzmann method (LBM).
Download Dataset:

Google drive link for the 3D normalized distance maps carbonate micro-CT images: https://drive.google.com/file/d/1G2HMvpO0z4J9GwioNVdElAW6B-nTVY5B/view?usp=sharing

Google drive link for the 3D binary maps carbonate micro-CT images: https://drive.google.com/file/d/1brAfMJFHogrkf842fn_DUb2lYNXV-cux/view?usp=sharing

Dataset Structure:

    Fine Resolution (FR) images
    Intermediate Resolution (IR) images
    Coarse Resolution (CR) images
    Permeability labels (mD)

After downloading, extract the dataset into the data/ directory.

Prerequisites

    Python 3.8 or higher
    CUDA-capable GPU (recommended)

📦 Dependencies

    PyTorch >= 1.13
    torchvision >= 0.13
    scikit-learn >= 1.0.2
    pandas >= 1.3.4
    numpy >= 1.21.5
    matplotlib >= 3.5.3
    tifffile >= 2021.7.2

Full list available in requirements.txt

## Transfer Learning Pipeline

The hierarchical transfer learning framework follows this sequence:

    Fine Resolution (FR): Train on high-resolution images
    Intermediate Resolution (IR): Transfer weights from FR model
    Coarse Resolution (CR): Transfer weights from IR model

Each script includes preprocessing, data loading, hyperparameter configuration, loss functions, model architecture, and training loops.

---

Thank you for using this repository! If you find it useful, please consider starring ⭐ the project on GitHub.
