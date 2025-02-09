import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn-v0_8-white')

from tensorflow.keras.utils import plot_model
import random
from PIL import Image
import glob
import seaborn as sns
from tqdm.notebook import tqdm
import time

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve,auc, accuracy_score, confusion_matrix, classification_report

import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

"""**Mounting Drive**"""

from google.colab import drive
drive.mount('/content/drive')

"""**Loading Dataset**"""

train_path = "/content/drive/MyDrive/Brain-Tumor-MRI-dataset/Training"
test_path = "/content/drive/MyDrive/Brain-Tumor-MRI-dataset/Testing"

classes = ['no_tumor','pituitary_tumor', 'meningioma_tumor','glioma_tumor']
#classes.index('no_tumor')

X_train = []
Y_train = []
#Function to load images and assign a class
def load_data(directory):
    for class_name in tqdm(os.listdir(directory),desc=directory):
        class_path = os.path.join(directory, class_name)
        class_label = class_name
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path)
            X_train.append(img)
            Y_train.append(classes.index(class_name))
    return X_train,Y_train
X_train , Y_train = load_data(train_path)
X_train , Y_train = load_data(test_path)

"""**Analysing and Pre-processing the Dataset**"""

# Function to count the number of images per class
def count_images_per_class(data, classes):
    count_per_class = {class_name: 0 for class_name in classes}
    for class_idx in data:
        class_name = classes[class_idx]
        count_per_class[class_name] += 1
    return count_per_class

# Count the number of images per class for both the training and testing datasets
train_count_per_class = count_images_per_class(Y_train, classes)

plt.figure(figsize=(8, 5))
#train data
train_bars = plt.bar(train_count_per_class.keys(), train_count_per_class.values(), color='#E17A8A')
plt.title('# Distribution of images by class (Training)')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
#Add the percentages
for bar in train_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height} ({height / np.sum(list(train_count_per_class.values())) * 100:.1f}%)', ha='center', va='bottom', fontsize=10)

plt.show()

def show_examples(X_data, Y_data, classes, num_examples=5):
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes, num_examples, figsize=(10, 8))

    for i, class_name in enumerate(classes):
        class_indices = [idx for idx, label in enumerate(Y_data) if label == i]
        axs[i, 0].set_title(class_name, fontsize=10, pad=10, fontweight='bold')
        for j in range(num_examples):
                img = X_data[class_indices[j]]
                axs[i, j].imshow(img)
                axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# Display examples of images for each class
show_examples(X_train, Y_train, classes)

# Most frequent image sizes
train_data_shapes = []
for img in X_train:
  train_data_shapes.append(img.shape)
# Count the occurrences for each size
shape_counts = {}
for shape in train_data_shapes:
  if shape not in shape_counts:
    shape_counts[shape] = 0
  shape_counts[shape] += 1
# Sort the shapes by number
sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
# Display the most frequent size
print("Most frequent Train images shapes:")
for shape, count in sorted_shapes[:3]:
  print(f"- {shape}: {count}")

shape = (225, 225, 3)

# Function to crop an image after preprocessing
def preprocess_and_crop_image(image):
    # Convert the image to grayscale
    if len(image.shape) == 3:  # Check if the image is in BGR format
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # if the image is already in grayscale
        gray = image.copy()

    # Convert the image from float32 (range [0, 1]) to uint8 (range [0, 255])
    if gray.dtype == np.float32:
        gray = np.uint8(gray * 255)  # Scale to the range [0, 255]

    # Apply histogram equalization to enhance contrast
    eq_gray = cv2.equalizeHist(gray)

    # Apply global thresholding
    _, global_thresh = cv2.threshold(eq_gray, 127, 255, cv2.THRESH_BINARY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(eq_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

    # Find contours in the image after adaptive thresholding
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contour found after thresholding.")
        return image  # Return the original image if no contours are found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Crop the image according to the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

image_path = "/content/drive/MyDrive/Brain-Tumor-MRI-dataset/Testing/glioma_tumor/image(67).jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Load and crop the image.
cropped_image = preprocess_and_crop_image(img)

# Display the original image and the cropped image.
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title("Original Image", fontsize=10)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Image (without empty spaces)", fontsize=10)
plt.axis('off')

plt.tight_layout()
plt.show()

# Example function to resize and normalize images
def preprocess_data(train_data, shape=(224, 224)):
    processed_data = []
    for img in tqdm(train_data):
        # Apply the cropping function
        cropped_img = preprocess_and_crop_image(img)

        # Resize the image
        resized_img = cv2.resize(cropped_img, (shape[1], shape[0]))

        # Normalize the image
        normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Add the preprocessed image to the processed data
        processed_data.append(normalized_img)

    return processed_data

# Preprocess the training data
X_train = preprocess_data(X_train)

# Convert to a NumPy array
X_train = np.array(X_train)

print(f"X_train shape: {np.shape(X_train)}")  # Display the shape of X_train

# Example function to display images from each class
def show_examples(X_data, Y_data, classes, num_examples=5):
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes, num_examples, figsize=(10, 8))

    for i, class_name in enumerate(classes):
        # Find the indices for the current class
        class_indices = [idx for idx, label in enumerate(Y_data) if label == i]

        # Ensure there are enough examples in the class
        num_class_examples = min(len(class_indices), num_examples)

        # Set the title for the class
        axs[i, 0].set_title(class_name, fontsize=10, pad=10, fontweight='bold')

        for j in range(num_class_examples):
            img = X_data[class_indices[j]]
            axs[i, j].imshow(img)
            axs[i, j].axis('off')

        # If there are fewer examples than `num_examples`, hide the remaining axes for that class.
        for j in range(num_class_examples, num_examples):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# Display example images for each class after preprocessing.
show_examples(X_train, Y_train, classes)

X_train, Y_train = shuffle(X_train,Y_train, random_state=64)

# Split the training set into training and testing sets.
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train, test_size=0.3,random_state=64)

# Divide the training set into training and test sets.
X_test,X_val,Y_test,Y_val = train_test_split(X_test,Y_test, test_size=2/3,random_state=64)

sizes = [len(Y_train), len(Y_val), len(Y_test)]
labels = ['Training', 'Validation', 'Test']
colors = ['#4285f4', '#E17A8A', '#fbbc05']

plt.figure(figsize=(6,3))
plt.rcParams.update({'font.size': 8})
patches, texts, autotexts = plt.pie(sizes,labels=labels,colors=colors, autopct='%.1f%%', explode=(0.1,0.08,0.08), shadow=True, startangle=60);
plt.title(f"Distribution of training and test images.")
plt.legend(patches, [f"{label} : {size}" for label, size in zip(labels, sizes)], loc="best")
plt.axis('Equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

Y_train =  np.array(tensorflow.keras.utils.to_categorical(Y_train))
Y_val =  np.array(tensorflow.keras.utils.to_categorical(Y_val))
Y_test =  np.array(tensorflow.keras.utils.to_categorical(Y_test))

# Define the data augmentation parameters.
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.2,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    x=np.array(X_train),  # Images
    y=np.array(Y_train) # Labels, converted to one-hot encoding.
)

#Define data augmentation parameters
random_index = np.random.randint(len(X_train))
image = X_train[random_index]
class_label_encoded = Y_train[random_index]
class_label = classes[np.argmax(class_label_encoded)]

# Display the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 5, 1)
plt.imshow(image)
plt.title('Original\n({})'.format(class_label), fontsize=8)
plt.axis('off')

# Generate and display augmented versions of the image
for i in range(4):
    augmented_image = train_datagen.random_transform(image)
    plt.subplot(1, 5, i + 2)
    plt.imshow(augmented_image)
    plt.title('Augmented Image {}'.format(i + 1), fontsize=8)
    plt.axis('off')

plt.show()

"""**Creating and Training the CNN Model**"""

CNN_model = Sequential()
CNN_model.add(Conv2D(16, (3,3), activation='relu', input_shape=shape))
CNN_model.add(MaxPooling2D(2,2))
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv2D(32, (3,3), activation='relu'))
CNN_model.add(MaxPooling2D(2,2))
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv2D(64, (3,3), activation='relu'))
CNN_model.add(MaxPooling2D(2,2))
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv2D(128, (3,3), activation='relu'))
CNN_model.add(MaxPooling2D(2,2))
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv2D(512, (3,3), activation='relu'))
CNN_model.add(MaxPooling2D(2,2))
CNN_model.add(Dropout(0.2))

CNN_model.add(Flatten())
CNN_model.add(Dense(512, activation='relu'))
CNN_model.add(Dropout(0.5))

CNN_model.add(Dense(len(classes), activation='softmax'))


# Compile the model
CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
CNN_model.summary()

tensorflow.keras.utils.plot_model(CNN_model,show_shapes=True, show_layer_names=True, dpi=50)

# Define the batch size
batch_size = 64
# Image shape: height, width, RBG
image_shape = shape
# The number of epochs
epochs = 80

print(f'Batch size: {batch_size}')
print(f'Image shape: {image_shape}')
print(f'Epochs: {epochs}')

# Add ReduceLROnPlateau, EarlyStopping, and checkpoint as callbacks during training
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_delta=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)  # "Stop training early if the validation performance does not improve for a certain number of epochs."
checkpoint = ModelCheckpoint(filepath='best_CNNModel.keras', monitor='val_accuracy', save_best_only=True, verbose=1)  # Save the best model in terms of validation performance.

# Train the model
import tensorflow as tf
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    print("GPU is available")
else:
    print("GPU is not available")

start_time = time.time()
CNN_history = CNN_model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    validation_data = (X_val, Y_val),
    callbacks=[reduce_lr, early_stop, checkpoint],
    verbose=1
)
end_time = time.time()
# End tracking the time
end_time = time.time()

# Calculate and print total training time
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

"""Evaluating the Model."""

CNN_history.history.keys()

tr_acc = CNN_history.history['accuracy']
tr_loss = CNN_history.history['loss']
tr_per = CNN_history.history['precision']
tr_recall = CNN_history.history['recall']
val_acc = CNN_history.history['val_accuracy']
val_loss = CNN_history.history['val_loss']
val_per = CNN_history.history['val_precision']
val_recall = CNN_history.history['val_recall']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]

Epochs = [i + 1 for i in range(len(tr_acc))]
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'


plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')

plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

# Evaluate the model on the training set
train_result = CNN_model.evaluate(X_train,Y_train)

# Display the performance metrics
print("Train Loss: {:.2f}%".format(train_result[0] * 100))
print("Train Accuracy: {:.2f}%".format(train_result[1] * 100))
print("Train Precision: {:.2f}%".format(train_result[2] * 100))
print("Train Recall: {:.2f}%".format(train_result[3] * 100))

validation_results_cnn = CNN_model.evaluate(X_val,Y_val)

print("Validation Loss: {:.2f}%".format(validation_results_cnn[0] * 100))
print("Validation Accuracy: {:.2f}%".format(validation_results_cnn[1] * 100))
print("Validation Precision: {:.2f}%".format(validation_results_cnn[2] * 100))
print("Validation Recall: {:.2f}%".format(validation_results_cnn[3] * 100))

"""Making Predictions using the CNN"""

predictions = CNN_model.predict(X_test)
y_true_test = np.argmax(Y_test, axis=1)
y_pred_test = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_true_test, y_pred_test)
print("Accuracy:{:.2f}%".format(accuracy * 100))

plt.figure(figsize=(6, 4))
heatmap = sns.heatmap(confusion_matrix(y_true_test,y_pred_test), annot=True, fmt='3g',
                      xticklabels=classes, yticklabels=classes, annot_kws={"size": 8})

plt.title('Confusion Matrix',fontsize=8)
plt.xlabel('Predicted Label',fontsize=8)
plt.ylabel('True Label',fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

class_report = classification_report(y_true_test, y_pred_test, target_names=classes)
print("Classification Report:")
print(class_report)

"""## EfficientNet-B0"""

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall

shape = (224, 224, 3)

base_model = EfficientNetB0(include_top=False,
                      input_shape=shape,
                      weights='imagenet')

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)

output_tensor = Dense(len(classes), activation='softmax')(x)

EfficientNet_model = Model(inputs=base_model.input, outputs=output_tensor)

EfficientNet_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy', Precision(), Recall()])

EfficientNet_model.summary()

reduce_lr_en = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_delta=0.0001, verbose=1)
#early_stop_en = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint_en = ModelCheckpoint(filepath='best_EfficientNetModel.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Check GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    print("GPU is available")
else:
    print("GPU is not available")

# Train the model
start_time = time.time()
EfficientNet_history = EfficientNet_model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    validation_data = (X_val, Y_val),
    callbacks=[reduce_lr_en, checkpoint_en],
    verbose=1
)
end_time = time.time()
# Calculate and print total training time
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

EfficientNet_history.history.keys()

tr_acc_en = EfficientNet_history.history['accuracy']
tr_loss_en = EfficientNet_history.history['loss']
val_acc_en = EfficientNet_history.history['val_accuracy']
val_loss_en = EfficientNet_history.history['val_loss']

index_loss_en = np.argmin(val_loss_en)
val_lowest_en = val_loss_en[index_loss_en]
index_acc_en = np.argmax(val_acc_en)
acc_highest_en = val_acc_en[index_acc_en]


Epochs = [i + 1 for i in range(len(tr_acc_en))]
loss_label_en = f'Best epoch = {str(index_loss_en + 1)}'
acc_label_en = f'Best epoch = {str(index_acc_en + 1)}'

plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')

plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss_en, 'r', label='Training loss')
plt.plot(Epochs, val_loss_en, 'g', label='Validation loss')
plt.scatter(index_loss_en + 1, val_lowest_en, s=150, c='blue', label=loss_label_en)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc_en, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc_en, 'g', label='Validation Accuracy')
plt.scatter(index_acc_en + 1, acc_highest_en, s=150, c='blue', label=acc_label_en)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

train_result_en = EfficientNet_model.evaluate(X_train,Y_train)

print("Train Loss: {:.2f}%".format(train_result_en[0] * 100))
print("Train Accuracy: {:.2f}%".format(train_result_en[1] * 100))
print("Train Precision: {:.2f}%".format(train_result_en[2] * 100))
print("Train Recall: {:.2f}%".format(train_result_en[3] * 100))

validation_results_en = EfficientNet_model.evaluate(X_test,Y_test)

print("Test Loss: {:.2f}%".format(validation_results_en[0] * 100))
print("Test Accuracy: {:.2f}%".format(validation_results_en[1] * 100))
print("Test Precision: {:.2f}%".format(validation_results_en[2] * 100))
print("Test Recall: {:.2f}%".format(validation_results_en[3] * 100))

predictions_en = EfficientNet_model.predict(X_test)
y_true_test = np.argmax(Y_test, axis=1)
y_pred_test_en = np.argmax(predictions_en, axis=1)

accuracy_en = accuracy_score(y_true_test, y_pred_test_en)
print("Accuracy:{:.2f}%".format(accuracy_en * 100))

plt.figure(figsize=(6, 4))
heatmap_en = sns.heatmap(confusion_matrix(y_true_test,y_pred_test_en), annot=True, fmt='3g',
                      xticklabels=classes, yticklabels=classes, annot_kws={"size": 8})

plt.title('Confusion Matrix',fontsize=8)
plt.xlabel('Predicted Label',fontsize=8)
plt.ylabel('True Label',fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

class_report_en = classification_report(y_true_test, y_pred_test_en, target_names=classes)
print("Classification Report:")
print(class_report_en)

from tabulate import tabulate
import pandas as pd

# Create a dictionary with model names and accuracies
data = {
    "Model": ["Custom CNN", "EfficientNet"],
    "Training Accuracy (%)": [train_result[1] * 100, train_result_en[1] * 100],
    "Validation Accuracy (%)": [validation_results_cnn[1] * 100, validation_results_en[1] * 100],
    "Test Accuracy (%)": [accuracy * 100, accuracy_en * 100],
}

# Convert the dictionary into a DataFrame
results_df = pd.DataFrame(data)

# Round the accuracy values to 2 decimal places
results_df = results_df.round({"Training Accuracy (%)": 2, "Validation Accuracy (%)": 2, "Test Accuracy (%)": 2})

# Print the table
print("Model Evaluation Results:")
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))
