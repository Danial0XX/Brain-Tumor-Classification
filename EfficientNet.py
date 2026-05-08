import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


train_dir = "Training"
test_dir = "Testing"
image_size = (224, 224)
batch_size = 32

print("Loading data for EfficientNet...")
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=image_size, batch_size=batch_size)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=image_size, batch_size=batch_size)
class_names = train_dataset.class_names




print("Building EfficientNetB0...")
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet',
    pooling='avg'
)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x) 
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


print("\nTRAINING EFFICIENTNET-B0 (3 Epochs)...")
history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)


print("\nCalculating metrics...")
y_true, y_pred_probs = [], []
for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images, verbose=0))

y_true = np.array(y_true)
y_pred = np.argmax(np.array(y_pred_probs), axis=1)

print("\n" + "="*30)
print("RESULTS (EfficientNetB0)")
print("="*30)
print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_true, y_pred, average='macro') * 100:.2f}%")
print(f"Recall:    {recall_score(y_true, y_pred, average='macro') * 100:.2f}%")
print(f"F-measure: {f1_score(y_true, y_pred, average='macro') * 100:.2f}%")
print("="*30)


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Bonus 1: EfficientNetB0 Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()