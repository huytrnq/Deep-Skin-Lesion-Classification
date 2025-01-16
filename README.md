# Deep Skin Lesion Classification
This a challenge to classify skin lesion, which is part of the course project for the Computer Aid Diagnosis course at the University of Girona.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Todos](#todos)
4. [Features](#features)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)

## Introduction
Skin cancer is the most common type of cancer, and early detection is crucial for successful treatment. The goal of this project is to develop a machine learning model for 2 challenging tasks: binary classification of benign and malignant skin lesions and three classes classification of Melanoma, Basal Cell Carcinoma, and Squamous Cell Carcinoma.

## Dataset
The dataset includes dermoscopic images in JPEG format with a distribution that reflects real-world settings, where benign cases outnumber malignant ones but with an overrepresentation of malignancies. The images come from:   
- HAM10000 Dataset (ViDIR Group, Medical University of Vienna)
- BCN_20000 Dataset (Hospital Clínic de Barcelona)
- MSK Dataset (ISBI 2017)  

The dataset consists of more than 15,000 images for binary classification and around 6,000 images for three classes classification. Data Distribution of two and three classes classification is shown below:

![Binary Classdistribution](./images/two_class_distribution.png)
<p align="center">
    Distribution of binary classes in the dataset.
</p>

![Three Classdistribution](./images/three_class_distribution.png)
<p align="center">
    Distribution of three classes in the dataset.
</p>

## Todos
- [x] Data Augmentation
- [x] mlflow monitoring
- [x] Model Training
- [x] Model Evaluation
- [x] Binary Classification
- [x] Three Classes Classification
- [x] Log augmentation to mlflow
- [x] Monitoring
- [x] TTA
- [x] Export Results on prediction
- [x] Cross Validation


## Features
1. **Data Augmentation**:
    - Geometric transformations (flip, rotate, scale, shift).
    - Contrast adjustment (CLAHE).
    - Noise and blur (Gaussian, elastic distortions).
2. **Model Tracking**:
    - Integrated **mlflow** for monitoring training, validation, and testing metrics.
    - Logging augmentations and cross-validation results.
3. **Test-Time Augmentation (TTA)**:
    - Enhanced inference by averaging predictions across augmented test samples.
4. **Cross-Validation**:
    - k-Fold cross-validation for robustness and generalization.

## Training

### Configuration
- **Model Architecture**: EfficientNet (b6), ResNet50, ResNeXt50 with pretrained weights from ImageNet.
- **Optimizer**: Adam optimizer.
- **Scheduler**: Cosine annealing learning rate scheduler.
- **Loss Function**: Weighted Cross-Entropy Loss.

### Monitoring
- All training logs, metrics, and configurations are tracked using **mlflow**.

### How to Train
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
    For Single Run:
   ```bash
   python exp.py --dataset <dataset_type> --epochs <number_of_epochs> --batch_size <batch_size> --lr <learning_rate> --optimizer <optimizer_name> --patience <patience_for_early_stopping> 
   ```

   For Cross Validation:
   ```bash
   python exp_cv.py --dataset <dataset_type> --epochs <number_of_epochs> --batch_size <batch_size> --lr <learning_rate> --optimizer <optimizer_name> --patience <patience_for_early_stopping> 
   ```


## Evaluation
### Binary Classification

| Model           | Configuration       | Input Size | Loss   | Accuracy | Kappa  |
|------------------|---------------------|------------|--------|----------|--------|
| **ResNet50**    | Affine & Brightness | 512        | 0.4430 | 0.9223   | 0.8445 |
| **ResNet50**    | Full Augmentation   | 512        | 0.3924 | 0.9233   | 0.8466 |
| **ResNext50**   | Full Augmentation   | 512        | 0.3568 | 0.9186   | 0.8371 |
| **ResNext50**   | Full Augmentation   | 384        | 0.3457 | 0.9249   | 0.8498 |
| **ResNext50**   | Full Augmentation   | 416        | 0.3638 | 0.9283   | 0.8540 |
| **EfficientNet B6** | Full Augmentation | 512        | 0.3307 | 0.9283   | 0.8566 |
| **EfficientNet B6** | Full Augmentation | 640        | 0.2975 | 0.9318   | 0.8634 |
| **EfficientNet B6** | Full Augmentation | 736        | **0.2877** | 0.9259   | 0.8519 |
| **EfficientNet B6** | Ensemble                   | -          | -      | **0.9391** | 0.8782 |
| **EfficientNet B6** | Augmentation + CV | 640        | -      | **0.9420** | **0.8840** |
| **EfficientNet B6** | Augmentation + CV | 736        | -      | 0.9407   | 0.8814 |

### Three Classes Classification

| Model           | Configuration       | Input Size | Loss   | Accuracy | Kappa  |
|------------------|---------------------|------------|--------|----------|--------|
| **ResNet50**    | Full Augmentation   | 512        | 0.1745 | 0.9559   | 0.9209 |
| **ResNet50**    | Full Augmentation   | 512        | 0.1423 | 0.9654   | 0.9375 |
| **ResNext50**   | Full Augmentation   | 512        | 0.1577 | 0.9693   | 0.9450 |
| **ResNext50**   | Full Augmentation   | 640        | **0.1413** | 0.9685   | 0.9434 |
| **EfficientNet B6** | Augmentation + CV | 512        | -      | 0.9709   | 0.9475 |
| **EfficientNet B6** | Augmentation + CV | 640        | -      | 0.9646   | 0.9360 |
| **EfficientNet B6** | Augmentation + CV | 736        | -      | **0.9827** | **0.9688** |

## Results
- The best model for binary classification is EfficientNet B6 with input size 736 and augmentation + CV, with an accuracy of 94.20% and a kappa of 0.8840.
- The best model for three classes classification is EfficientNet B6 with input size 736 and augmentation + CV, with an accuracy of 98.27% and a kappa of 0.9688.
- Results on test set: 6 341 images for binary classification and 2 122 images for three classes classification.
    | Binary Classification | Three Classes Classification |
    |------------------------|-------------------------------|
    | Accuracy: 0.957        | Kappa score: 0.952            |



