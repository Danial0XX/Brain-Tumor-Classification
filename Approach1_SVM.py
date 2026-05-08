import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


train_dir = "Training"
test_dir = "Testing"
image_size = (224, 224)
batch_size = 32

print("Loading data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=image_size, batch_size=batch_size)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=image_size, batch_size=batch_size)
class_names = train_dataset.class_names

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))


print("Loading MobileNetV1...")
feature_extractor = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)


def extract_features(dataset):
    features, labels = [], []
    for images, batch_labels in dataset:
        batch_features = feature_extractor.predict(images, verbose=0)
        features.append(batch_features)
        labels.append(batch_labels.numpy())
    return np.vstack(features), np.concatenate(labels)

print("Extracting features from Training set (This takes a minute)...")
X_train, y_train = extract_features(train_dataset)

print("Extracting features from Testing set...")
X_test, y_test = extract_features(test_dataset)


print("Training the Support Vector Machine (SVM)...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


print("Calculating final metrics...")
y_pred = svm_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


print("\n" + "="*30)
print("APPROACH 1 RESULTS (MobileNet + SVM)")
print("="*30)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F-measure: {f1 * 100:.2f}%")
print("="*30)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Approach 1: DL Features + SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()