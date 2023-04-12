# import streamlit as st
# import numpy as np
# from keras.datasets import cifar10
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.optimizers import Adam
# from keras.utils import to_categorical
# import cv2

# # Load the CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Convert and preprocess CIFAR-10 images
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # Convert CIFAR-10 labels to one-hot encoding
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

# # Function to apply rotation to images
# def apply_rotation(image, angle):
#     rows, cols, _ = image.shape
#     M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#     rotated_image = cv2.warpAffine(image, M, (cols, rows))
#     return rotated_image

# # Function to straighten rotated image based on predicted rotation label
# def straighten_image(image, rotation_label):
#     rotation_angle = 0
#     if rotation_label == 1:
#         rotation_angle = 90
#     elif rotation_label == 2:
#         rotation_angle = -90
#     elif rotation_label == 3:
#         rotation_angle = 180
#     straightened_image = apply_rotation(image, rotation_angle)
#     return straightened_image

# # Create a rotation prediction model
# def create_rotation_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     return model

# # Generate rotated CIFAR-10 images
# def generate_rotated_dataset():
#     rotated_images = []
#     rotated_labels = []
#     for i in range(len(x_test)):
#         image = x_test[i]
#         for angle in [90, -90, 180]:
#             rotated_image = apply_rotation(image, angle)
#             rotated_images.append(rotated_image)
#             rotated_labels.append(y_test[i])
#     rotated_images = np.array(rotated_images)
#     rotated_labels = np.array(rotated_labels)
#     return rotated_images, rotated_labels

# # Compile the rotation prediction model
# rotation_model = create_rotation_model()
# rotation_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the rotation prediction model only if it hasn't been trained before
# if 'rotation_model_trained' not in st.session_state:
#     st.session_state['rotation_model_trained'] = False

# if not st.session_state['rotation_model_trained']:
#     rotated_images, rotated_labels = generate_rotated_dataset()
#     rotation_model.fit(rotated_images, rotated_labels, batch_size=64, epochs=15)
#     st.session_state['rotation_model_trained'] = True

# # Streamlit web application
# st.title('CIFAR-10 Rotation Prediction')
# st.subheader('Select an image from the CIFAR-10 test set')

# # Display original image
# image_index = st.slider('Image Index', 0, len(x_test)-1, 0)
# original_image = x_test[image_index]
# st.image(original_image, caption='Original Image', use_column_width=True)

# # Get predicted rotation label and probabilities
# rotation_label_probabilities = rotation_model.predict(np.expand_dims(original_image, axis=0))[0]
# rotation_label = np.argmax(rotation_label_probabilities)
# rotation_probabilities = ', '.join([f'{i}: {p:.4f}' for i, p in enumerate(rotation_label_probabilities)])

# # Display predicted rotation label and probabilities
# st.write(f'Predicted Rotation Label: {rotation_label}')
# # st.write(f'Rotation Probabilities: {rotation_probabilities}')

# # Apply rotation to image based on predicted rotation label
# straightened_image = straighten_image(original_image, rotation_label)
# st.image(straightened_image, caption='Straightened Image', use_column_width=True)

import gc
gc.collect()

from keras.backend import clear_session
clear_session()

import streamlit as st
import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Load CIFAR dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define angles for rotation
angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

# Function to rotate an image
def rotate_image(image, angle):
    rotated_image = image.rotate(angle)
    return np.array(rotated_image)

# Function to straighten rotated image based on predicted rotation label
def straighten_image(image, rotation_label):
    if rotation_label == 0:
        rotation_angle = -30
    elif rotation_label == 1:
        rotation_angle = -60
    elif rotation_label == 2:
        rotation_angle = -90
    elif rotation_label == 3:
        rotation_angle = -120
    elif rotation_label == 4:
        rotation_angle = -150
    elif rotation_label == 5:
        rotation_angle = -180
    elif rotation_label == 6:
        rotation_angle = -210
    elif rotation_label == 7:
        rotation_angle = -240
    elif rotation_label == 8:
        rotation_angle = -270
    elif rotation_label == 9:
        rotation_angle = -300                                                
    straightened_image = rotate_image(Image.fromarray(image), rotation_angle)
    return straightened_image

# Rotate training images
x_train_rotated = []
y_train_rotated = []
for i, j in enumerate(y_train):
    rotated_image = rotate_image(Image.fromarray(x_train[i]), angles[j[0]])
    x_train_rotated.append(rotated_image)
    y_train_rotated.append(j)

x_train_rotated = np.array(x_train_rotated)
y_train_rotated = np.array(y_train_rotated)

# Rotate test images
x_test_rotated = []
y_test_rotated = []
for i, j in enumerate(y_test):
    rotated_image = rotate_image(Image.fromarray(x_test[i]), angles[j[0]])
    x_test_rotated.append(rotated_image)
    y_test_rotated.append(j)

x_test_rotated = np.array(x_test_rotated)
y_test_rotated = np.array(y_test_rotated)

# Convert labels to categorical
y_train_rotated = to_categorical(y_train_rotated, num_classes=10)
y_test_rotated = to_categorical(y_test_rotated, num_classes=10)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the rotation prediction model only if it hasn't been trained before
if 'rotation_model_trained' not in st.session_state:
    st.session_state['rotation_model_trained'] = False

if not st.session_state['rotation_model_trained']:
    model.fit(x_train_rotated, y_train_rotated, batch_size=32, epochs=20)
    st.session_state['rotation_model_trained'] = True

# Streamlit web application
st.title('CIFAR-10 Rotation Prediction')
st.subheader('Select an image from the CIFAR-10 test set')

# Display original image
image_index = st.slider('Image Index', 0, len(x_test_rotated)-1, 0)
original_image = x_test_rotated[image_index]
st.image(original_image, caption='Original Image', use_column_width=True)

# Get predicted rotation label and probabilities
rotation_label_probabilities = model.predict(np.expand_dims(original_image, axis=0))[0]
rotation_label = np.argmax(rotation_label_probabilities)

# Apply rotation to image based on predicted rotation label
straightened_image = straighten_image(original_image, rotation_label)
st.image(straightened_image, caption='Straightened Image', use_column_width=True)