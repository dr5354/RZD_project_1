# Object Detection on Railway Images

This directory contains the code for training and inference of the model in the **Object detection**.
## Model Architecture

In this case, we use the **Faster RCNN** architecture. Backbone(ResNet50FPN) is pre-trained on the ImageNet dataset.

## Steps to Prevent Overfitting

Several strategies are implemented to prevent overfitting:

* **Data Augmentation:** 
* **Regularization:** Gradient clipping is used to prevent gradient explosion.
* **Training Optimization:**

    Adam with different learning rates (LR) for different parts of the model.
    Learning rate reduction on plateau.
    Validation
    Early stopping
* **Class Imbalance Handling:** The function `weighted_loss` has been introduced to calculate weights based on the frequency of classes in dataset annotations.

## Inference


This code provides inference of the trained model. To download the weights of the model, follow the link: [https://drive.google.com/file/d/1JaKeJFAlUtW3f9QSGvAjiJwbmirO1P26/view?usp=drive_link]