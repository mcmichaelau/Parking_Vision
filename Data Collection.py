# 1. Loop through library of parking lot images
# 2. Detect objects and do instance segmentation, record coordinates
# 3. Mark ground truth location (center of parking space), record coordinates
# 4. Compile all data into training dataset with parameters and target values
# 5. Normalize dimensions

import cv2
import os
import torch
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

directory = 'data collection images'


class CollectData(object):
    def __init__(self):

        self.image_coordinates = 0

        self.collected_data = pd.DataFrame(columns=['ul', 'ur', 'Output'])

        for filename in os.listdir(directory):

            f = os.path.join(directory, filename)

            img = cv2.imread(f)

            result = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            box_coords = result.pandas().xyxy[0].drop(columns=['confidence', 'class'])

            for index, line in box_coords.iterrows():

                self.ul = [int(line['xmin']), int(line['ymin'])]
                self.ur = [int(line['xmax']), int(line['ymax'])]

                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.extract_coordinates)

                new_img = cv2.imread(f)

                cv2.circle(new_img, self.ul, 7, (0, 0, 255), -1)
                cv2.circle(new_img, self.ur, 7, (0, 0, 255), -1)

                cv2.imshow('image', new_img)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

            os.remove(f)

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [x, y]
            self.collected_data = self.collected_data.append({'ul': self.ul, 'ur': self.ur,
                                                              'Output': self.image_coordinates},
                                                             ignore_index=True)
            print(self.collected_data)


CollectData()

# for coord in box_coords:
#
#     cv2.circle(img, coord, 7, (0, 0, 255), -1)

#
# cv2.imshow('image', result)
#
# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()

# import os
#
# # File name
# file = 'file.txt'
#
# # File location
# location = "/home/User/Documents"
#
# # Path
# path = os.path.join(location, file)
#
# # Remove the file
# # 'file.txt'
# os.remove(path)
