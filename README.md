<h1 align="center">🌿 Plant Disease Detection Using Convolutional Neural Networks</h1>

<p align="center">
Automated tomato leaf disease classification — comparing a custom CNN against ResNet50 and MobileNetV2 transfer learning.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

---

## 📖 Overview

This project implements and compares **three deep learning models** for automated tomato leaf disease detection using the **PlantVillage** dataset. Manual diagnosis of crop diseases is slow, expensive, and subjective — this project engineers an automated diagnostic framework using CNNs to address that gap.

Three approaches are trained and benchmarked: a **Custom CNN** built from scratch (baseline), and two **transfer learning** models — **ResNet50** (depth and accuracy) and **MobileNetV2** (efficiency for deployment).

---

## 🎯 Research Questions

| # | Research Question |
|---|---|
| **RQ1** | How effective is a Custom CNN as a baseline for tomato disease classification? |
| **RQ2** | How much does Transfer Learning (ResNet50, MobileNetV2) improve performance? |
| **RQ3** | What is the efficiency trade-off between accuracy and inference speed? |

---

## 📊 Results

All three models were evaluated on a held-out test set of 1,500 unseen images.

| Model | Test Accuracy | Macro F1 | Model Size | Inference / image (GPU) |
|-------|:-------------:|:--------:|:----------:|:-----------------------:|
| Custom CNN (baseline) | 84.87% | 0.85 | 220.6 MB | 65.1 ms |
| **ResNet50** | **96.80%** | **0.97** | 206.9 MB | 111.7 ms |
| MobileNetV2 | 89.07% | 0.89 | **24.6 MB** | 122.5 ms |

### Key Findings

- **ResNet50** was the most accurate model at **96.80%**, a ~12-point jump over the custom baseline — clearly demonstrating the value of transfer learning (RQ2).
- The **Custom CNN** established a solid 84.87% baseline from scratch, confirming the task is learnable without pretrained weights (RQ1).
- **MobileNetV2** offered the best efficiency trade-off (RQ3): nearly 9× smaller than ResNet50 (24.6 MB vs 206.9 MB) while still reaching 89% accuracy — the strongest candidate for mobile/field deployment.
- **Grad-CAM** visualizations confirmed the models focus on actual diseased leaf regions rather than background, validating that predictions are based on meaningful features.

---

## 🗂️ Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Images:** 10,000 (balanced — 1,000 per class)
- **Classes (10):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, and Healthy.
- **Split:** 70% train / 15% validation / 15% test (stratified)

> The full dataset is not included in this repo due to size. Download it from the link above to reproduce the results.

---

## ⚙️ Methodology

1. **Data Exploration** — Verified all 10 classes are present and balanced; visualized samples.
2. **Preprocessing** — Stratified 70/15/15 split; images resized to 224×224; per-model normalization (Min-Max for the Custom CNN, model-specific `preprocess_input` for ResNet50 and MobileNetV2).
3. **Augmentation** — Rotation, horizontal flip, zoom, and width/height shifts on the training set.
4. **Custom CNN** — Built with the Keras Functional API; stacked Conv2D/MaxPooling blocks (32→64→128→...), trained up to 25 epochs with Adam and categorical cross-entropy.
5. **Transfer Learning** — ResNet50 and MobileNetV2 (ImageNet weights) with a two-phase strategy: train a new classification head with the base frozen, then fine-tune the top 30 layers at a low learning rate (1e-5).
6. **Training callbacks** — EarlyStopping, ModelCheckpoint (best val accuracy), and ReduceLROnPlateau.
7. **Evaluation** — Accuracy, precision, recall, F1 (per-class and weighted), confusion matrices, inference time, and model size.
8. **Interpretability** — Grad-CAM heatmaps to verify the regions driving predictions.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **Models:** Custom CNN, ResNet50, MobileNetV2 (transfer learning)
- **Data/ML:** NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Interpretability:** Grad-CAM
- **Environment:** Google Colab (GPU)

---

## 🚀 How to Run

This project was developed in **Google Colab** with GPU acceleration.

1. Open `plantdiseasedetection.ipynb` in Google Colab.
2. Download the PlantVillage dataset (link above) and place it in your Google Drive.
3. Update the dataset path in the setup cell to match your Drive location.
4. Run the cells top to bottom — the notebook handles extraction, preprocessing, training all three models, evaluation, comparison charts, and Grad-CAM.

> **Trained models:** the `.keras` model files are large and not stored in this repo. They are regenerated when you run the training cells. *(Optionally add a download link here if you host them on Drive.)*

---

## 🔮 Future Improvements

- Extend beyond tomato to multi-crop disease detection
- Deploy MobileNetV2 in a Streamlit or mobile app for in-field use
- Test on real-world (non-PlantVillage) field images to assess generalization
- Explore newer efficient architectures (EfficientNet, ViT) for the accuracy/size trade-off

---

## 👤 Author

**Gul Sher Khan**
Software Engineering Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gul-sher-khan-0119aa264/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:glshrrkhan@gmail.com)
