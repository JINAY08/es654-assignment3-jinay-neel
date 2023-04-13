import streamlit as st
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10

# loading CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# defining angles for rotation
angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

# function for rotating an image
def rotate_image(image, angle):
    rotated_image = image.rotate(angle)
    return np.array(rotated_image)

# function for straightening the rotated image based on predicted label
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

# rotating training images
x_train_rotated = []
y_train_rotated = []
for i, j in enumerate(y_train):
    rotated_image = rotate_image(Image.fromarray(x_train[i]), angles[j[0]])
    x_train_rotated.append(rotated_image)
    y_train_rotated.append(j)

x_train_rotated = np.array(x_train_rotated)
y_train_rotated = np.array(y_train_rotated)

# rotating test images
x_test_rotated = []
y_test_rotated = []
for i, j in enumerate(y_test):
    rotated_image = rotate_image(Image.fromarray(x_test[i]), angles[j[0]])
    x_test_rotated.append(rotated_image)
    y_test_rotated.append(j)

x_test_rotated = np.array(x_test_rotated)
y_test_rotated = np.array(y_test_rotated)

# converting labels to categorical
y_train_rotated = to_categorical(y_train_rotated, num_classes=10)
y_test_rotated = to_categorical(y_test_rotated, num_classes=10)

# CNN model for training
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

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the rotation prediction model only if it hasn't been trained before
if 'rotation_model_trained' not in st.session_state:
    st.session_state['rotation_model_trained'] = False

if not st.session_state['rotation_model_trained']:
    model.fit(x_train_rotated, y_train_rotated, batch_size=32, epochs=20)
    st.session_state['rotation_model_trained'] = True

# Streamlit web application
st.title('CIFAR-10 Rotation Prediction')
st.subheader('Select an image from the CIFAR-10 test set')

# displaying the rotated test image
image_index = st.slider('Image Index', 0, len(x_test_rotated)-1, 0)
original_image = x_test_rotated[image_index]
st.image(original_image, caption='Original Image', use_column_width=True)

# obtaining predicted label and probabilities
rotation_label_probabilities = model.predict(np.expand_dims(original_image, axis=0))[0]
rotation_label = np.argmax(rotation_label_probabilities)

# applying rotation to image based on predicted label to straighten the image
straightened_image = straighten_image(original_image, rotation_label)
st.image(straightened_image, caption='Straightened Image', use_column_width=True)
