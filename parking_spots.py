import cv2
import pickle

img = cv2.imread("parking lot still.png")

roi = cv2.selectROI("select the area", img)

# Crop image
cropped_image = img[int(roi[1]):int(roi[1] + roi[3]),
                int(roi[0]):int(roi[0] + roi[2])]

# Display cropped image
cv2.imshow("Cropped image", cropped_image)
cv2.waitKey(0)

print(roi)
