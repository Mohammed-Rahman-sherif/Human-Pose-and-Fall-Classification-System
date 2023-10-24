import cv2

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Capture frames from a video source (replace '0' with your video source)
cap = cv2.VideoCapture('Fall-Detection-Data\FALL-SIDE\Data3-Fall-side-3-Depth.avi')  # 0 represents the default camera

while True:
    ret, frame = cap.read()

    b, g, r = cv2.split(frame)

    # Apply adaptive thresholding to each channel
    thresholded_b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

    # Merge the thresholded channels to create an RGB image
    thres = cv2.merge((thresholded_b, thresholded_g, thresholded_r))

    # Apply background subtraction to detect the moving part
    fg_mask = bg_subtractor.apply(thres)

    # Apply the mask to the moving object
    moving_part = cv2.bitwise_and(thres, thres, mask=fg_mask)

    # Display only the moving object
    cv2.imshow('Moving Object', thres)

    if cv2.waitKey(100) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
