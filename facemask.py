# Mask Detection System written in Python utilizing Tensorflow at Shell Hacks Hackathon 2023. Our group created an application
# where users upload an image and it is determined whether or not they are wearing a mask. Here is a link to a website we created
# using React.js with more information! https://aimaskdetection.netlify.app/


from zipfile import ZipFile
dataset = '/content/face-mask-dataset.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

with_mask_files = os.listdir('/content/data/with_mask')
print(with_mask_files[0:5])
print(with_mask_files[-5:])

without_mask_files = os.listdir('/content/data/without_mask')
print(without_mask_files[0:5])
print(without_mask_files[-5:])

# create the labels

withMaskLabels = [1]*3725

withoutMaskLabels = [0]*3828

print(len(withMaskLabels))
print(len(withoutMaskLabels))

labels = withMaskLabels + withoutMaskLabels #total amount of masks

print(len(labels))
print(labels[0:5])
print(labels[-5:])

# displaying with mask image
img = mpimg.imread('/content/data/with_mask/with_mask_829.jpg')
imgplot = plt.imshow(img)
plt.show()

# displaying without mask image
img = mpimg.imread('/content/data/without_mask/without_mask_2925.jpg')
imgplot = plt.imshow(img)
plt.show()

# convert images to numpy arrays+

with_mask_path = '/content/data/with_mask/'

data = []

for img_file in with_mask_files:

  image = Image.open(with_mask_path + img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)



without_mask_path = '/content/data/without_mask/'


for img_file in without_mask_files:

  image = Image.open(without_mask_path + img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)

  # converting image list and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train_scaled = X_train/255

X_test_scaled = X_test/255

num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# training the neural network, check here if not working
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)

loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)

h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()

# Input image path
input_image_path = '/content/IMG_6513.jpg'  # Update with the path to an image with a mask

# Load the input image
input_image = cv2.imread(input_image_path)

# Display the input image using cv2_imshow
from google.colab.patches import cv2_imshow
cv2_imshow(input_image)

# Assuming you have a loaded and compiled model, make sure it's correctly configured for mask detection.
# Load and compile your model here (replace 'model' with your actual model)
# model = ...

# Resize and scale the input image
input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255.0  # Scale pixel values to the [0, 1] range

# Reshape the input image for prediction (assuming your model expects shape [1, 128, 128, 3])
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

# Make the prediction using your model
input_prediction = model.predict(input_image_reshaped)

# Assuming a binary classification problem (mask or no mask)
# If your model outputs probabilities, you can use a threshold to decide the class
threshold = 0.5  # Adjust this threshold as needed

if input_prediction[0][0] >= threshold:
    print('The person in the image is wearing a mask')
else:
    print('The person in the image is not wearing a mask')

temp = input('Path of the image to be predicted: ')
input_image_path = temp

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resized = cv2.resize(input_image, (128,128))

input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

input_prediction = model.predict(input_image_reshaped)

print(input_prediction)


input_pred_label = np.argmax(input_prediction)

print(input_pred_label)


if input_pred_label == 0:

  print('The person in the image is wearing a mask')

else:

  print('The person in the image is not wearing a mask')

