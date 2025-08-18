# Multiclass-Fish-Image-Classification-deep-learning

This project is a deep learning solution for classifying fish images into multiple categories. It uses Convolutional Neural Networks (CNNs) and transfer learning with several pre-trained models to achieve high accuracy. The final model is deployed as an interactive web application using Streamlit, allowing users to upload an image and get a real-time prediction.

#  Key Features

- Multiclass Classification: Classifies images of fish into various species.

- Transfer Learning: Compares the performance of a custom CNN model against several pre-trained       architectures, including:

   - VGG16

   - ResNet50

   -  MobileNetV2

   - InceptionV3

    - EfficientNetB0

- Data Augmentation: Enhances model robustness by applying techniques like rotation, zoom, and        flipping to the training data.

- Performance Evaluation: Compares model performance using metrics such as accuracy, precision,       recall, F1-score, and a confusion matrix.

- Interactive Deployment: A user-friendly web application built with Streamlit where users can        upload an image and get an instant prediction.

#  Technologies Used

Category	              Technology/Library

Language	              Python

Frameworks	           TensorFlow, Keras, Streamlit

Key Libraries	        OpenCV, NumPy, scikit-learn

Model Architectures	  VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0

# Project Structure

The repository contains the following key files and folders:

 - README.md: Project description and instructions.

 - fish_classification_*.ipynb: Jupyter notebooks for each model, demonstrating data preprocessing,    training, and evaluation.

 - fish.py: The Python script for the Streamlit web application.

 - models/: (Optional) Folder to store the trained .h5 model files.

# How to Run the App
1. Clone the Repository:

git clone https://github.com/JOTHILAKSHMI-H/Multiclass-Fish-Image-Classification-deep-learning.git

cd Multiclass-Fish-Image-Classification-deep-learning

2. Install Dependencies:

  - Ensure you have Python and pip installed.

  - Install the required libraries:

pip install -r requirements.txt
(Note: You will need a requirements.txt file listing all the libraries. You can generate one with pip freeze > requirements.txt.)

3. Run the Streamlit Application:

   - Make sure your trained model file (best_model.h5 or similar) is in the correct path.

   - Execute the following command:

streamlit run fish.py

   - The app will open automatically in your web browser, ready for use.

 # Model Evaluation
 
The project evaluates each model's performance to identify the best one for deployment. The key metrics considered include:

  - Accuracy: The proportion of correct predictions.

  - Precision: The accuracy of positive predictions.

  - Recall: The ability to find all positive samples.

  - F1-Score: The harmonic mean of precision and recall.

   - Confusion Matrix: A table showing the performance of the model on the test data, providing a visual summary of where the model is making errors.
