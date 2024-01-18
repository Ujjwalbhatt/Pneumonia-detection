import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directories for training, testing, and validation sets
train_dir = r'D:\Pneumonia detection\chest_xray\chest_xray\train'
test_dir = r'D:\Pneumonia detection\chest_xray\chest_xray\test'
val_dir = r'D:\Pneumonia detection\chest_xray\chest_xray\val'

# Data augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Load the VGG16 model, pre-trained on ImageNet data
base_model = VGG16(input_shape=(150,150,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the layers

# Create a new model on top
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # adjust based on the size of your dataset
    epochs=7,
    validation_data=val_generator,
    validation_steps=50)  # adjust based on the size of your dataset

# Evaluate the model on the test set
print(model.evaluate(test_generator))
# Save the entire model to a HDF5 file
model.save('pneumonia_detection_model.h5')
print("Model saved successfully!")