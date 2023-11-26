import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model('best_models\lr-1e-05_opti-RMSprop\epoch138-val_acc0.95-val_loss0.14.h5')

# Open the video file
video_path = 'Fall-Detection-Data\standing\Data2-ADL-grasp-3-standing.avi'
cap = cv2.VideoCapture(video_path)

# Define the frame rate for frame extraction
frame_rate = 1  # Adjust as needed

# Initialize an empty list to store predictions
classes = {0: 'ADL-GRASP', 1: 'ADL-LAY', 2: 'ADL-SIT', 3: 'ADL-WALK', 4: 'FALL-BACK', 5: 'FALL-ENDUPSIT', 6: 'FALL-FRONT', 7: 'FALL-SIDE', 8: 'standing'}

predictions = []
your_width = 64
your_height = 64

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop when we reach the end of the video

    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (your_width, your_height))  # Resize to match model input size
    frame = frame.astype('float32') / 255.0  # Normalize pixel values

    # Make a prediction for the frame using the loaded model
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    prediction = model.predict(frame)

    # Process the prediction as needed (e.g., decode class labels)
    # In this example, we assume a classification model with class labels

    # Decode the prediction into human-readable labels
    decoded_label = np.argmax(prediction, axis=-1)  # Assumes class index is used

    # Append the decoded label to the predictions list
    predictions.append(decoded_label)

# Close the video capture
cap.release()

# Process the predictions as needed (e.g., count occurrences)
# You can also display or save the results here

# Example: Count the occurrences of predicted labels
label_counts = {}
for prediction in predictions:
    label = prediction[0]  # Assuming single class label per frame
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1

print("Label Counts:")
for label, count in label_counts.items():
    print(f"Class {label}: {count} >>> Class Label: {classes[label]}")
