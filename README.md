# 🧠 Brain Tumor MRI Classification: Feature Extraction vs. End-to-End DL

## 📌 Project Overview
This repository contains the implementation of a comparative study evaluating two different Deep Learning paradigms for classifying Brain Tumors from MRI scans. The project was developed as part of the CS460 Deep Learning coursework.

The study compares a hybrid **DL-Based Feature Extraction + ML Classifier** approach against a fully integrated **End-to-End Deep Learning** model.

## 🗄️ Dataset
* **Domain:** Medical Imaging (Diagnostic Classification)
* **Dataset:** Brain Tumor MRI Dataset (7,200 images)
* **Classes (4):** Glioma, Meningioma, Pituitary Tumor, No Tumor
* **Preprocessing:** Images resized to 224x224 and pixel normalized (rescaled by 1./255).

## ⚙️ Methodology & Architectures
Both primary approaches utilized **MobileNetV1**, selected specifically for its depthwise separable convolutions, making it highly efficient for potential deployment in resource-constrained medical environments (e.g., edge devices).


1. **Approach 1 (Hybrid):** MobileNetV1 used strictly as a fixed feature extractor (`include_top=False`), feeding continuous feature vectors into a Linear Support Vector Machine (SVM).
2. **Approach 2 (End-to-End):** MobileNetV1 adapted with a custom 4-class dense classification head and trained iteratively.

Note: As an extended bonus analysis, EfficientNetB0 was also evaluated using the end-to-end pipeline to study the effects of optimized network scaling on this dataset.

## 📊 Performance Results

| Metric | Approach 1 (SVM Hybrid) | Approach 2 (End-to-End DL) |
| :--- | :--- | :--- |
| **Accuracy** | 92.75% | 93.00% |
| **Precision** | 93.06% | 93.47% |
| **Recall** | 92.75% | 93.00% |
| **F-measure** | 92.64% | 92.91% |

### 💡 Key Insights
* **Speed vs. Accuracy:** The SVM hybrid model (Approach 1) trained in a fraction of the time, making it ideal for rapid edge deployment. The End-to-End model (Approach 2) achieved slightly higher accuracy by fine-tuning weights specific to tumor morphology, but at a higher computational cost.
Network Complexity (Bonus): Testing the modern EfficientNetB0 architecture yielded exceptional results (93.12% test accuracy). It completely avoided the severe overfitting issues commonly seen when applying massive, older models to datasets of this size, proving that mathematically optimized architectural scaling is highly effective for medical imagery.

## 🚀 How to Run
1. Clone this repository.
2. Download the Brain Tumor MRI dataset from Kaggle ([click here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)) and place the `Training` and `Testing` directories in the root folder.
3. Install dependencies: `pip install tensorflow scikit-learn matplotlib seaborn`
4. Execute the Python scripts sequentially.
