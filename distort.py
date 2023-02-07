import pafy
import cv2
import numpy as np
import math

mPafy = pafy.new('https://www.youtube.com/watch?v=U7HRKjlXK-Y&ab_channel=Supercircuits')

mStream = mPafy.getbest(preftype="mp4")

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# display video with spaces drawn
capture = cv2.VideoCapture(mStream.url)

ret, frame = capture.read()

frame = cv2.imread('Screen Shot 2022-09-20 at 1.27.05 PM.png')

vid_dim = frame.shape

print(f'vid dim: {vid_dim}')

img = 'parking lot 3.jpg'

img = cv2.imread(img)
dimensions = img.shape

print(f'photo dim: {dimensions}')

coordinates = [[426, 866], [1057, 838], [438, 1033], [1260, 985]]


def separate_coordinates(coordinates):
    i = 0

    vertex_coordinates = []

    for coordinate in coordinates:

        i = i + 1
        if i % 4 == 0:
            vertex_coordinate = coordinates[i - 4:i]

            vertex_coordinates.append(vertex_coordinate)

    return vertex_coordinates


def crop_and_warp(frame, vertex_coordinates):

    for shape in vertex_coordinates:

        corner_ul = shape[0]
        corner_ur = shape[1]
        corner_ll = shape[2]
        corner_lr = shape[3]

        n_ul = [0, 0]
        n_ur = [math.sqrt(((corner_ur[0]) ** 2) + ((corner_ur[1]) ** 2)), 0]
        n_ll = [0, math.sqrt(((corner_ll[0]) ** 2) + ((corner_ll[1]) ** 2))]
        n_lr = [math.sqrt(((corner_ur[0]) ** 2) + ((corner_ur[1]) ** 2)),
                math.sqrt(((corner_ll[0]) ** 2) + ((corner_ll[1]) ** 2))]

        # Locate points of the documents
        # or object which you want to transform
        pts1 = np.float32(shape)
        pts2 = np.float32([n_ul, n_ur, n_ll, n_lr])

        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (int(n_lr[0]), int(n_lr[1])))

        # Wrap the transformed image
        cv2.imshow('frame1', result)  # Transformed Capture

        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()


vertex_coordinates = separate_coordinates(coordinates)

crop_and_warp(frame, vertex_coordinates)
