# Author: Mohammed Rahman Sherif
# Date: 2023-10-07
# Time: 15:55:11

####################################
# Total input data: 43674
# Train size: 12951
####################################

import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

X = []
y = []
your_width = 64
your_height = 64

main_folder = 'Fall-Detection-Data'  

class_folders = os.listdir(main_folder)

# Create a dictionary to map class labels to integers
classes = {'ADL-GRASP': 0, 'ADL-LAY': 1, 'ADL-SIT': 2, 'ADL-WALK': 3, 'FALL-BACK': 4, 'FALL-ENDUPSIT': 5, 'FALL-FRONT': 6, 'FALL-SIDE': 7, 'standing': 8}
i = 0
# Iterate through each class folder
for class_folder in class_folders:
    class_path = os.path.join(main_folder, class_folder)

    # List video files in the class folder
    video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]

    # Iterate through each video file in the class folder
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)

        # Initialize a VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
        else:
            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                # Break the loop if we have reached the end of the video
                if not ret:
                    break

                b, g, r = cv2.split(frame)

                # Apply adaptive thresholding to each channel
                thresholded_b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
                thresholded_g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
                thresholded_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

                # Merge the thresholded channels to create an RGB image
                frame = cv2.merge((thresholded_b, thresholded_g, thresholded_r))

                # Resize the frame to your desired size
                img_size = (your_width, your_height)  # Replace with the desired width and height
                frame = cv2.resize(frame, img_size)
                frame = frame.astype('float32')

                # Normalize the pixel values to be in the range [0, 1]
                frame = frame / 255.0

                # Append the frame to X
                X.append(frame)

                # Assign the class label to y based on the class folder name
                y.append(class_folder)
                i += 1
                print("Number of images processed: ", i)

        # Release the VideoCapture object
        cap.release()

# Convert the lists to NumPy arrays
X = np.array(X)
y = np.array(y)
print('X Shape', X.shape)
print('Y Shape', y.shape)

# Map class labels to integers for y_train and y_test
y_int = [classes[label] for label in y]

# Convert the integer labels to one-hot encoded format
y_encoded = to_categorical(y_int, num_classes=9)

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.10, random_state=33, shuffle=True)
print('X_train Shape', X_train.shape)
print('X_test Shape', X_test.shape)
# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(your_width, your_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(9, activation='softmax')
])

# Compile the model
learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and TensorBoard callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)
fold_path = f'lr-{learning_rate}_opti-RMSprop_AdapThres/'
os.mkdir(f'best_models/{fold_path}')
save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"best_models/{fold_path}/epoch{{epoch:02d}}-val_acc{{val_accuracy:.2f}}-val_loss{{val_loss:.2f}}.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
print('Step 6 of 8: Model Training Started.')
# Train the model
history = model.fit(X_train, y_train_encoded, epochs=1000, batch_size=512, validation_data=(X_test, y_test_encoded), callbacks=[early_stopping, tensorboard_callback, save_model_callback])
print('Step 7 of 8: Model Training complete.')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
print('Step 8 of 8: Training Visualization complete.')
print('All Stages complete. Thank You!.')
