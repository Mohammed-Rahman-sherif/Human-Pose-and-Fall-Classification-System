import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('best_models\lr-1e-05_opti-RMSprop_AdapThres\epoch159-val_acc0.91-val_loss0.24.h5')

# Set up video capture from a webcam (you can also specify a video file)
cap = cv2.VideoCapture(0)

classes = {0: 'ADL-GRASP', 1: 'ADL-LAY', 2: 'ADL-SIT', 3: 'ADL-WALK', 4: 'FALL-BACK', 5: 'FALL-ENDUPSIT', 6: 'FALL-FRONT', 7: 'FALL-SIDE', 8: 'standing'}

while True:
    ret, frame = cap.read()
    original_frame = frame.copy()

    if not ret:
        break
    
    b, g, r = cv2.split(frame)

    # Apply adaptive thresholding to each channel
    thresholded_b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

    # Merge the thresholded channels to create an RGB image
    thres = cv2.merge((thresholded_b, thresholded_g, thresholded_r))
    cv2.imshow('thres', thres)

    frame = cv2.resize(thres, (64, 64))
    
    # Preprocess the frame
    frame = frame.astype('float32') / 255.0
    
    # Make predictions
    prediction = model.predict(np.expand_dims(frame, axis=0))
    predicted_class = np.argmax(prediction)
    # Display the predicted class on the frame
    cv2.putText(original_frame, f'Predicted: Class {classes[predicted_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Canny Video', original_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
