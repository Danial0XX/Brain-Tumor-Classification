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

*Note: As an extended bonus analysis, ResNet50 was also evaluated using the end-to-end pipeline to study the effects of network depth and overfitting on this dataset.*

## 📊 Performance Results

| Metric | Approach 1 (SVM Hybrid) | Approach 2 (End-to-End DL) |
| :--- | :--- | :--- |
| **Accuracy** | 92.75% | 93.00% |
| **Precision** | 93.06% | 93.47% |
| **Recall** | 92.75% | 93.00% |
| **F-measure** | 92.64% | 92.91% |

### 💡 Key Insights
* **Speed vs. Accuracy:** The SVM hybrid model (Approach 1) trained in a fraction of the time, making it ideal for rapid edge deployment. The End-to-End model (Approach 2) achieved slightly higher accuracy by fine-tuning weights specific to tumor morphology, but at a higher computational cost.
* **Network Complexity:** Testing the heavier ResNet50 architecture resulted in severe overfitting (99% training accuracy vs 25% validation), proving that the lightweight MobileNetV1 was the superior architectural choice for this specific dataset size.

## 🚀 How to Run
1. Clone this repository.
2. Download the Brain Tumor MRI dataset from Kaggle and place the `Training` and `Testing` directories in the root folder.
3. Install dependencies: `pip install tensorflow scikit-learn matplotlib seaborn`
4. Execute the Python scripts sequentially.
