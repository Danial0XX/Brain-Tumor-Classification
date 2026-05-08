import tensorflow as tf


train_dir = "Training" 
test_dir = "Testing"


train_dataset_raw = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  image_size=(224, 224), 
  batch_size=32
)


test_dataset_raw = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size=(224, 224),
  batch_size=32
)


class_names = train_dataset_raw.class_names
print("Classes found:", class_names)


normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset_raw.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset_raw.map(lambda x, y: (normalization_layer(x), y))

print("Data preprocessing complete! Ready to build the model.")