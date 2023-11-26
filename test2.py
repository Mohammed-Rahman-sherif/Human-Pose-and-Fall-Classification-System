import cv2
import numpy as np

cap = cv2.VideoCapture('Fall-Detection-Data\FALL-BACK\Data11-Fall-back-3-Depth.avi')
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    _, video = cam.read()
    
    if not ret:
        cap = cv2.VideoCapture('Fall-Detection-Data\FALL-BACK\Data11-Fall-back-3-Depth.avi')
        continue

    Mov_vid = frame.copy()

    bb, gg, rr = cv2.split(frame)

    # Apply adaptive thresholding to each channel
    thresholded_bb = cv2.adaptiveThreshold(bb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_gg = cv2.adaptiveThreshold(gg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_rr = cv2.adaptiveThreshold(rr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

    # Merge the thresholded channels to create an RGB image
    image = cv2.merge((thresholded_bb, thresholded_gg, thresholded_rr))

    cv2.imshow('Original', image)
    ##############################################################################################
    b, g, r = cv2.split(video)

    # Apply adaptive thresholding to each channel
    thresholded_b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
    thresholded_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

    # Merge the thresholded channels to create an RGB image
    thresholded_image = cv2.merge((thresholded_b, thresholded_g, thresholded_r))


    cv2.imshow('OTSU Edges', thresholded_image)
    ##############################################################################################
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
