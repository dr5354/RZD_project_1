# Object Detection on Railway Images

This directory contains the code for training and inference of the model in the **Object detection**.


## Model Architecture

Thus, we use **Retinanet (ResNet50FPN)** as the initial architecture. The model is pre-trained on the ImageNet dataset.

## Steps to Prevent Overfitting

Several steps have been taken to address this issue, including:

* **Data Augmentation:** 
* **Regularization:** Focal loss is also used to solve the problem of class imbalance.
* **Training Optimization:** 

        Adam with different learning rates (LR) for different parts of the model
        Learning rate reduction on plateau
        Validation
        Early stopping

## Inference

This code provides inference of the trained model. To download the weights of the model, follow the link: [https://drive.google.com/file/d/1mnCuc2GIgul9jd5wVV2PATxcNe2yjImX/view?usp=drive_link]
