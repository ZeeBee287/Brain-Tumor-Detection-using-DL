# 🧠 Brain Tumor Detection Using Deep Learning

---

## 🎯 Motivation

Brain tumors pose a significant global health challenge, impacting individuals across all age groups. Early and precise diagnosis is critical for effective treatment planning and enhancing patient outcomes. The variability in tumor appearance and the complexity of brain structures make manual examination prone to errors. The availability of advanced datasets and the opportunity to contribute to healthcare and interdisciplinary research inspired this project.

## 🎯 Objectives

1. 🚀 Enhance the efficiency and accuracy of deep learning (DL) models for brain tumor classification.
2. 📊 Compare multiple DL models to determine the optimal approach for real-world application.

## 🧪 Methodology

### 📁 Dataset
The dataset, sourced from **Kaggle**. It comprises 3,264 MRI scans of brain tumors categorized as:
- Glioma Tumors (926 images)
- Meningioma Tumors (937 images)
- Pituitary Tumors (901 images)
- No Tumors (500 images)

### 🛠️ Preprocessing
Steps taken to prepare the data for model training:
1. ✂️ **Cropping**: Removed irrelevant regions to focus on tumor areas
2. 📏 **Resizing**: Standardized image size to **225×225×3**
3. ⚖️ **Normalization**: Scaled pixel values to a range of **\[0, 1]**

### 🔄 Data Augmentation
To boost dataset diversity and reduce overfitting:
* 🔄 Rotation (±30°)
* 🪞 Shear (0.2)
* 🔍 Zoom (0.05)
* ↔️ Horizontal Flip

### 🤖 Model Training
Two models were employed:
1.	Custom CNN: 
    - 5 convolutional layers with increasing filters: 16 → 32 → 64 → 128 → 512
    - MaxPooling2D, ReLU activations, and dropout layers (rates: 0.2 and 0.5)
    - Fully connected layer with 512 units and softmax output for classification

2.	EfficientNet-B0: 
    - Advanced scalable architecture for efficient training and high accuracy

### ⚙️ Training Parameters
- Batch size: 64
- Epochs: 30
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam

## 📈 Results

**EfficientNet-B0** outperformed the custom CNN with:

* ✅ **Validation Accuracy**: **97.55%**
* 📉 **Validation Loss**: **0.10**
* 🧪 **Training Accuracy**: 99.74%

It provided a superior balance between performance and computational efficiency.

## 🧾 Conclusion

This project highlights the potential of **deep learning**, especially **EfficientNet-B0**, in automating brain tumor classification from MRI scans. Such systems can assist radiologists by:

* 🩺 Increasing diagnostic accuracy
* 🧑‍⚕️ Reducing workload
* 🕒 Accelerating treatment timelines

## Contributors

- **Alishba Zulfiqar**
- **Bushra Tanveer**
- **Zahra Batool**
