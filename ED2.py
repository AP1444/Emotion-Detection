import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2



# Load the trained model
model = load_model('/Users/anshpanwar/Desktop/Python project/emotion_detection_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
plt.show()

# Define a function to preprocess the image
def preprocess_image_for_prediction(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=2) 
    image = np.expand_dims(image, axis=0) 
    return image

# Define a function to predict emotion
def predict_emotion(image_path):
    processed_image = preprocess_image_for_prediction(image_path)
    prediction = model.predict(processed_image)
    print(f'Raw Prediction: {prediction}')
    emotion_index = np.argmax(prediction)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Surprise', 'Sad', 'Happy', 'Neutral']
    predicted_emotion = emotion_labels[emotion_index]
    return predicted_emotion


# Example prediction
image_path = "/Users/anshpanwar/Desktop/Python project/images.jpeg"
predicted_emotion = predict_emotion(image_path)
print(f'Predicted Emotion: {predicted_emotion}')