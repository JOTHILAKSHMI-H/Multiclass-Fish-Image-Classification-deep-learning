# Multiclass-Fish-Image-Classification-deep-learning

This project focuses on classifying fish images into multiple categories using Convolutional Neural Networks (CNNs) and transfer learning with pre-trained model
Train a CNN model from scratch for multiclass fish classification and Use pre-trained models (Transfer Learning) like ResNet50, VGG16, InceptionV3, EfficientNetB0, MobileNetV2.

### Data Preprocessing and Augmentation
Rescale images to [0, 1] range.
Apply data augmentation techniques like rotation, zoom, and flipping to enhance model robustness.

### Model Training
Train a CNN model from scratch.
Experiment with five pre-trained models (e.g., VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).

### Fine-Tuning
Apply transfer learning techniques on pre-trained models.  
Fine-tune parameters to maximize accuracy on the fish dataset.

### Model Saving
Save the best-performing model in `.h5` or `.pkl` format for future inference an

### Model Evaluation
Compare metrics such as accuracy, precision, recall, F1-score, and confusion matrix across all models.


### Deployment Process:
A Streamlit web application was built to make the fish classification model interactive and user-friendly.
- Upload fish images from their device

- Get predictions for the fish category

- View the model's confidence scores for each class

