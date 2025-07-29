# 🧠 Brain Tumor Classification AI

This project uses a deep learning model to classify brain MRI images into four types of brain tumors. It leverages transfer learning with the **VGG16** architecture and deploys the model using **FastAPI** for web access.

---
---

## 🗂 Dataset

The dataset used in this project contains MRI images classified into four categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, and **No Tumor**.

📁 **Download from Kaggle**:  
[Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- Total Images: ~3,264
- Format: JPG
- Organized into subfolders by class


##  Model Highlights

- **Model**: Pretrained VGG16 (fine-tuned)
- **Classes**:
  - Glioma Tumor
  - Meningioma Tumor
  - Pituitary Tumor
  - No Tumor
- **Framework**: TensorFlow / Keras
- **Input Size**: 224x224 RGB images
- **Accuracy**: High performance on validation data

---

## 🌐 Web Deployment with FastAPI

The model is deployed via a FastAPI backend with a simple HTML frontend. Users can:

- Upload an MRI image
- Get an instant prediction of the tumor type

To run the FastAPI app locally:

```bash
uvicorn main:app --reload
