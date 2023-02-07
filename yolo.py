import torch
import pafy
import cv2
import subprocess as sp
import numpy as np
import math
import pandas as pd
from IPython.display import display

mPafy = pafy.new('https://www.youtube.com/watch?v=c38K8IsYnB0&ab_channel=suuwebcam4')

mStream = mPafy.getbest(preftype="mp4")

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# display video with spaces drawn
capture = cv2.VideoCapture(mStream.url)

ret, frame = capture.read()

# coordinates

coordinates = [[9, 409], [93, 423], [33, 548], [6, 549], [103, 422], [205, 423], [158, 549], [32, 547], [206, 432],
               [316, 436], [294, 554], [157, 556], [317, 431], [439, 430], [442, 557], [298, 561], [438, 432],
               [562, 426], [594, 550], [440, 554], [556, 427], [674, 421], [733, 548], [589, 554], [662, 424],
               [782, 409], [862, 533], [734, 546], [769, 408], [886, 414], [975, 526], [863, 540], [885, 419],
               [974, 413], [1070, 517], [970, 531], [974, 410], [1049, 399], [1152, 507], [1067, 517], [1048, 404],
               [1125, 407], [1210, 487], [1148, 506]]


# separate coordinates into each shape
def separate_coordinates(coordinates):
    i = 0

    n = 0

    vertex_coordinates = []

    for coordinate in coordinates:

        i = i + 1

        if i % 4 == 0:
            n = n + 1
            vertex_coordinate = coordinates[i - 4:i], 'available'

            vertex_coordinates.append(vertex_coordinate)

    df = pd.DataFrame(vertex_coordinates, columns=['Coordinates', 'status'])

    # display(df)

    return df


# Crop parking spaces


# def crop_and_warp(frame, vertex_coordinates):
#     cropped_frames = []
#
#     for shape in vertex_coordinates:
#         corner_ul = shape[0]
#         corner_ur = shape[1]
#         corner_ll = shape[2]
#         corner_lr = shape[3]
#
#         n_ul = [0, 0]
#         n_ur = [math.sqrt(((corner_ur[0]) ** 2) + ((corner_ur[1]) ** 2)), 0]
#         n_ll = [0, math.sqrt(((corner_ll[0]) ** 2) + ((corner_ll[1]) ** 2))]
#         n_lr = [math.sqrt(((corner_ur[0]) ** 2) + ((corner_ur[1]) ** 2)),
#                 math.sqrt(((corner_ll[0]) ** 2) + ((corner_ll[1]) ** 2))]
#
#         # Locate points of the documents
#         # or object which you want to transform
#         pts1 = np.float32(shape)
#         pts2 = np.float32([n_ul, n_ur, n_ll, n_lr])
#
#         # Apply Perspective Transform Algorithm
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
#         result = cv2.warpPerspective(frame, matrix, (int(n_lr[0]), int(n_lr[1])))
#
#         cropped_frames.append(result)
#
#     return cropped_frames


def car_coords(frame):
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    box_coords = results.pandas().xyxy[0].drop(columns=['confidence', 'class'])

    midline_coords = []

    for index, line in box_coords.iterrows():

        if line['name'] == 'car' or line['name'] == 'bus' or line['name'] == 'truck':
            midline_coord = (int((line['xmax'] + line['xmin']) / 2), int((line['ymax'] + line['ymin']) / 2))

            bottom_coord = (int((line['xmax'] + line['xmin']) / 2),
                            int(((line['ymax'] + line['ymin']) / 2) + ((line['ymax'] - line['ymin']) / 2)))

            circle_coord = (int((line['xmax'] + line['xmin']) / 2),
                            int(((line['ymax'] + line['ymin']) / 2) + ((line['ymax'] - line['ymin']) * (1 / 4))))

            # midline_coords.append(midline_coord)

            midline_coords.append(circle_coord)

            # midline_coords.append(bottom_coord)

    return midline_coords

    # for point in midline_coords:
    #     print(point)
    #     cv2.circle(frame, tuple(point), 5, (0, 0, 255))
    #
    # cv2.imshow('Test image', frame)
    #
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()


# def placement_detection(car_coordinates, vertex_coordinates):
#     # for point in car_coordinates:
#     #     print(point)
#     #     cv2.circle(frame, tuple(point), 20, (0, 0, 255))
#
#     # cv2.imshow('Test image', frame)
#     #
#     # if cv2.waitKey(0) & 0xFF == ord('q'):
#     #     cv2.destroyAllWindows()
#
#     for shape in vertex_coordinates:
#         for coordinate in car_coordinates:
#
#             x1, y1 = shape[0]
#             x2, y2 = shape[1]
#             x3, y3 = shape[2]
#             x4, y4 = shape[3]
#
#             top_left_x = min([x1, x2, x3, x4])
#             top_left_y = min([y1, y2, y3, y4])
#
#             contour_1 = [x1 - top_left_x, y1 - top_left_y]
#             contour_2 = [x2 - top_left_x, y2 - top_left_y]
#             contour_3 = [x3 - top_left_x, y3 - top_left_y]
#             contour_4 = [x4 - top_left_x, y4 - top_left_y]
#
#             cv2.circle(frame, tuple(contour_1), 5, (0, 0, 255))
#
#             cv2.imshow('test', frame)
#
#             if cv2.waitKey(0) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#
#             # check if circle is in bounding box
#
#             contour = np.array([contour_1, contour_2, contour_3, contour_4])
#
#             print(contour)
#
#             result = cv2.pointPolygonTest(contour, coordinate, False)
#
#             print(result)


def check_space(frame, car_points, dataframe):

    for index, row in dataframe.iterrows():
        shape = np.array(row['Coordinates'])

        for point in car_points:

            check = cv2.pointPolygonTest(shape, point, False)

            if check == 1:
                row['status'] = 'taken'
                # print(f'space {index}:taken')

            # else:
            #     row['status'] = 'available'

            cv2.circle(frame, tuple(point), 5, (0, 0, 255))

    display(dataframe)

    # for point in car_points:
    #
    #     space_num = 0
    #
    #     for item in dataframe:

            #
            # space_num = item[0]
            #
            # shape = item[1]
            #
            # shape = np.array(shape)
            #
            # check = cv2.pointPolygonTest(shape, point, False)
            #
            # if check == 1:
            #     print(f'space {space_num}:taken')
            #
            # cv2.circle(frame, tuple(point), 5, (0, 0, 255))-----


        # cv2.imshow('Test image', frame)
        #
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()


# def crop_and_cover(frame, vertex_coordinates):
#     finished_frames = []
#
#     for shape in vertex_coordinates:
#
#         ret, frame = capture.read()
#
#         # print(shape)
#
#         x1, y1 = shape[0]
#         x2, y2 = shape[1]
#         x3, y3 = shape[2]
#         x4, y4 = shape[3]
#
#         top_left_x = min([x1, x2, x3, x4])
#         top_left_y = min([y1, y2, y3, y4])
#         bot_right_x = max([x1, x2, x3, x4])
#         bot_right_y = max([y1, y2, y3, y4])
#
#         cropped = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
#
#         # print(cropped.shape)
#         #
#         # print(f'tl:{x1 - top_left_x}, {y1 - top_left_y}')
#         # print(f'tl:{x2 - top_left_x}, {y2 - top_left_y}')
#         # print(f'tl:{x3 - top_left_x}, {y3 - top_left_y}')
#         # print(f'tl:{x4 - top_left_x}, {y4 - top_left_y}')
#
#         contour_1 = [x1 - top_left_x, y1 - top_left_y]
#         contour_2 = [x2 - top_left_x, y2 - top_left_y]
#         contour_3 = [x3 - top_left_x, y3 - top_left_y]
#         contour_4 = [x4 - top_left_x, y4 - top_left_y]
#
#         contours = np.array([contour_1, contour_2, contour_3, contour_4])
#
#         mask = np.zeros(cropped.shape, dtype=np.uint8)
#         cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
#
#         # apply the mask
#         masked_image = cv2.bitwise_and(cropped, mask)
#
#         # cv2.imshow('img', masked_image)
#
#         results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#         labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
#
#         box_coords = results.pandas().xyxy[0].drop(columns=['confidence', 'class'])
#
#         midline_coords = []
#
#         for index, line in box_coords.iterrows():
#
#             if line['name'] == 'car' or line['name'] == 'bus' or line['name'] == 'truck':
#                 midline_coord = (int((line['xmax'] + line['xmin']) / 2), int(line['ymax']))
#
#                 midline_coords.append(midline_coord)
#
#         for point in midline_coords:
#             shape = np.array(shape)
#
#             check = cv2.pointPolygonTest(shape, point, False)
#
#             # print(f'shape: {shape}')
#             # print(f'shape type: {type(shape)}')
#
#             if check == 1:
#                 print(f'space:{check}')
#
#             cv2.circle(frame, tuple(point), 5, (0, 0, 255))
#
#         cv2.imshow('Test image', frame)
#
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()


dataframe = separate_coordinates(coordinates)

car_coordinates = car_coords(frame)

check_space(car_points=car_coordinates, frame=frame, dataframe=dataframe)

# crop_and_cover(frame, vertex_coordinates)

# placement_detection(car_coordinates, vertex_coordinates)

# crop_and_cover(frame, vertex_coordinates)
#
# while True:
#
#     ret, frame = capture.read()
#
#     cropped_frames = crop_and_warp(frame, vertex_coordinates)
#
#     for frame in cropped_frames:
#
#         cv2.imshow('frame', frame)
#
#         size = frame.shape
#         print(f'size:{size}')
#
#         results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         results.print()
#
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#
#         print('------ new frame -------')
#
#
#
#
#
#
#
#
#
#
#     print(i)
#
#     i = i+1
#
#     # Inference
#     results = model(frame)
#     results.print()
#     results.show()
#     # or .show(), .save()
#
#     # cv2.imshow('yt', frame)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
