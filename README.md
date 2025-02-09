## Brain Tumor Detection Using Deep Learning
---
### Motivation
Brain tumors pose a significant global health challenge, impacting individuals across all age groups. Early and precise diagnosis is critical for effective treatment planning and enhancing patient outcomes. The variability in tumor appearance and the complexity of brain structures make manual examination prone to errors. The availability of advanced datasets and the opportunity to contribute to healthcare and interdisciplinary research inspired this project.

### Objectives
1.	Enhance the efficiency and accuracy of a deep learning (DL) model for brain tumor classification.
2.	Compare the performance of different DL models to determine the optimal approach.

### Methodology
Dataset
The dataset used in this project was sourced from Kaggle. It comprises 3,264 MRI scans of brain tumors categorized as:
- Glioma Tumors (926 images)
- Meningioma Tumors (937 images)
- Pituitary Tumors (901 images)
- No Tumors (500 images)

### Preprocessing
To prepare the data for analysis:
1.	Cropping: Removed irrelevant spaces to emphasize critical features.
2.	Resizing: Standardized image dimensions to 225×225×3.
3.	Normalization: Scaled pixel values between 0 and 1 to improve model convergence.

### Data Augmentation
Applied to the training data to increase diversity:
- Rotations (±30∘) 
- Shearing (0.2)
- Zoom (0.05)
- Horizontal flipping

### Model Training
Two models were employed:
1.	Custom CNN: 
    - 5 convolutional layers with increasing filters (16, 32, 64, 128, 512)
    - MaxPooling2D, ReLU activations, and dropout layers (rates: 0.2 and 0.5)
    - Fully connected layer with 512 units and softmax output for classification

2.	EfficientNet-B0: 
    - Advanced scalable architecture for efficient training and high accuracy

### Training Parameters:
- Batch size: 64
- Epochs: 30
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam

### Results
EfficientNet-B0 achieved the highest validation accuracy of 97.55% and the lowest validation loss of 0.10. It exhibited balanced training accuracy (99.74%) and validation accuracy, outperforming the Custom CNN in both performance and computational efficiency.

### Conclusion
This project demonstrates the potential of CNN-based models, particularly EfficientNet-B0, in accurately classifying brain tumors from MRI images. By automating this critical diagnostic task, the system aims to enhance diagnostic precision, reduce radiologists' workload, and accelerate patient treatment timelines.
