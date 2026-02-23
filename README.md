# RaabinWBC-DenseNet201-MixedAttention

Official implementation of **DenseNet201 with Mixed Spatial–Channel Attention (SAM + CAM)** for high-precision white blood cell classification on the **Raabin-WBC dataset**.

This repository accompanies the manuscript:

**Refining DenseNet201 with Mixed Attention for High-Precision White Blood Cell Classification**  
> Submitted to *The Visual Computer*

If you use this code, please cite the associated manuscript (see Citation section below).

## Overview

White Blood Cells (WBCs) are classified into five types:

- Basophil  
- Eosinophil  
- Lymphocyte  
- Monocyte  
- Neutrophil  

Based on morphology, they can also be grouped into:

- **Agranulocytes**: Lymphocyte, Monocyte  
- **Granulocytes**: Neutrophil, Eosinophil, Basophil  

This implementation introduces a **Mixed Attention refinement strategy** applied to DenseNet201:

- Spatial Attention Module (SAM)
- Channel Attention Module (CAM)

The model is trained in a **multi-task setting**:

1. 5-class WBC classification
2. Binary classification (Agranulocyte vs Granulocyte)

---

##  Dataset Structure

The code expects the following folder layout:

Raabin-WBC/
Basophil/
Eosinophil/
Lymphocyte/
Monocyte/
Neutrophil/

Each folder must contain image files (`.png`, `.jpg`, `.jpeg`, etc.).

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
Training

Run training with:

python train.py --data_dir "/path/to/Raabin-WBC" --epochs 10 --batch_size 16

Outputs are saved in:

outputs/
 ├── best_model.keras
 ├── training_log.csv
 ├── splits/
 └── figures/
Evaluation

To evaluate a trained model:

python eval.py --data_dir "/path/to/Raabin-WBC" --model_path outputs/best_model.keras

Evaluation includes:

5-class accuracy

5-class classification report

5-class confusion matrix

Binary accuracy (Agranulocytes vs Granulocytes)

Binary classification report

Binary confusion matrix

Reproducibility

This repository ensures reproducibility by:

Fixing random seeds

Using stable class ordering

Saving dataset split indices

Logging hyperparameters

Saving the best-performing model

Saved split files are located in:

outputs/splits/

These files can be archived on Zenodo to guarantee full experimental reproducibility.

Model Architecture

Backbone:

DenseNet201 (ImageNet pre-trained)

Attention Modules:

Spatial Attention Module (SAM)

Channel Attention Module (CAM)

Classifier:

Shared feature head

Two output branches (5-class + binary)

Citation

If you use this repository, please cite:

Salam Jabbar et al.
Refining DenseNet201 with Mixed Attention for High-Precision White Blood Cell Classification.
Submitted to The Visual Computer.

(Full citation will be updated upon publication.)
