import cv2
import pafy
import cv2
import numpy as np
import math

mPafy = pafy.new('https://www.youtube.com/watch?v=U7HRKjlXK-Y&t=72s&ab_channel=Supercircuits')

mStream = mPafy.getbest(preftype="mp4")


class DrawLineWidget(object):
    def __init__(self):
        capture = cv2.VideoCapture(mStream.url)

        ret, self.original_image = capture.read()

        # self.original_image = cv2.imread('full parking lot still.png')

        self.corner_coordinates = []

        cv2.namedWindow('Select Parking Spots')
        cv2.setMouseCallback('Select Parking Spots', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [x, y]
            print(self.image_coordinates)
            self.corner_coordinates.append(self.image_coordinates)

    def show_image(self):
        return self.original_image


if __name__ == '__main__':
    draw_line_widget = DrawLineWidget()
    while True:
        cv2.imshow('Select Parking Spots', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            print(draw_line_widget.corner_coordinates)
            cv2.destroyAllWindows()
            exit(1)
