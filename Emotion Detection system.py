from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

global img
choice = int(input('How do you want to check: \n ' + '1. On live Photo \n' + ' 2. On stored photo(should be edited in code)\n '))
if(choice == 1):
# Accessing the camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read() 
        cv2.imshow('frame', frame) # Display the resulting frame

        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to take a photo
            cv2.imwrite('photo.jpg', frame) # Save the photo as 'photo.jpg'
            break
    cap.release()
    cv2.destroyAllWindows()
    # Load the captured image
    img = cv2.imread('photo.jpg')
    # plt.imshow(img)
    # plt.show()
    plt.imshow(img[:,:,::-1])
    plt.show()


elif(choice == 2):
    img =  cv2.imread('/Users/anshpanwar/Desktop/Python project/images.jpeg')  
    # plt.imshow(img)
    # plt.show()
    plt.imshow(img[:,:,::-1])
    plt.show()

else:
    print('Invalid choice')
    
obj = DeepFace.analyze(img, actions=['emotion'])
# print(obj)
dominant_emotion = obj[0]['dominant_emotion']
print("Dominant Emotion:", dominant_emotion)