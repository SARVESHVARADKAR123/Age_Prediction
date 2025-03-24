# Age Detection using Fine-Tuned CNN

## Overview
This project fine-tunes a pre-trained CNN model (MobileNetV2) for age detection using the **UTKFace dataset**. The model predicts the age of a person based on their facial image and is evaluated based on the **Mean Absolute Error (MAE)**.

## Dataset
- **UTKFace Dataset**: A collection of facial images labeled with age, gender, and ethnicity.
- Each image filename contains the age, gender, and ethnicity information.

## Features
- Fine-tunes **MobileNetV2** (or ResNet50) for age regression.
- Uses data augmentation techniques to improve generalization.
- Evaluates performance using **Mean Absolute Error (MAE)**.
- Saves the trained model for future inference.

## Dataset Preparation
- The UTKFace dataset contains images labeled with age.
- Images are resized and normalized before feeding them into the model.
- The dataset is split into training and validation sets for better evaluation.

## Model Training
- A pre-trained CNN model is fine-tuned for age detection.
- Fully connected layers are added on top of the base model.
- The model is trained on the dataset and optimized for accuracy.

## Model Evaluation
- The trained model is evaluated based on **Mean Absolute Error (MAE)**.
- A lower MAE indicates better performance in predicting age.

## Prediction
- The trained model can predict the age of a person based on a facial image.
- The image is preprocessed before making a prediction.

## Results
- The model achieves reasonable accuracy in predicting age from images.
- The performance can be improved with further fine-tuning and additional training data.

## Resources
- **Google Drive Links:**
  - **Trained Model (.keras):** [Download Here](https://drive.google.com/file/d/1idrrFDUTuYforM5WHzP4_b4mnG7-ILKv/view?usp=drive_link)
  - **Colab Notebook:** [Open Here](https://colab.research.google.com/drive/13euswQlPrZcYFMMRkTvq20RrDW2GP7Sj?usp=drive_link)
  - **Dataset (UTKFace):** [Download Here](https://www.kaggle.com/datasets/jangedoo/utkface-new)
  - **Pre-trained Weights:** [Download Here](https://drive.google.com/file/d/1Jr8JuI5QsZM9qYd12WUEV3anbTlhtunG/view?usp=drive_link)

