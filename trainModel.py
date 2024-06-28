from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv('/Users/anshpanwar/Desktop/Python project/fer2013.csv')

def preprocess_image(image, target_size=(48, 48)):
    image = image.reshape(target_size)
    image = image.astype('float32') / 255.0
    return image

X = []
y = []
for index, row in data.iterrows():
    pixels = row['pixels'].split(' ')
    image = np.array(pixels, dtype='float32')
    image = preprocess_image(image)
    X.append(image)
    y.append(row['emotion'])

X = np.array(X)
X = X.reshape(X.shape[0], 48, 48, 1)
y = to_categorical(y, num_classes=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Convolutional Neural Networks(CNN) model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save('emotion_detection_model.h5')