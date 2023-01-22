import cv2 as cv
from rescale import *
import mediapipe as mp
import pyautogui as py
py.FAILSAFE = False

hand_detector = mp.solutions.hands.Hands()
drawing_utilities = mp.solutions.drawing_utils
screen_width, screen_height = py.size()
cap = cv.VideoCapture(0)
index_y=0
def read():
    global index_y
    while True:
        isTrue, frame = cap.read()
        frame= cv.flip(frame,1)  #flipping the frame on the y axis; taki lateral inversion hatt jaye
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        output_frame = hand_detector.process(rgb_frame)
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])

        hands=output_frame.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utilities.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    x=int(landmark.x*frame_width)
                    y=int(landmark.y*frame_height)

                    if id == 8:
                        cv.circle(img=frame, center=(x, y),radius=15, color=(0,255 , 255))
                        index_x = screen_width/frame_width *x
                        index_y = screen_height / frame_height * y
                        #print(index_x, index_y)
                        py.moveTo(index_x, index_y)  # moves using the PyAutoGui package

                    #print(index_y)
                    if id == 4:
                        cv.circle(img=frame, center=(x, y),radius=15, color=(0,255 , 255))
                        thumb_x = screen_width/frame_width *x
                        thumb_y = screen_height / frame_height * y
                         #print("checker",check,"Thumb coordinates:   ", index_y,thumb_y)
                        print("outside click region   ", abs(index_y-thumb_y))
                        if abs(index_y-thumb_y) < 50:
                            print("CLICK")
                            py.click()
                            py.sleep(1)

        rescaled_frame = rescaleFrame(frame, 2)
        cv.imshow('Virtual Mouse', rescaled_frame)
        stop()
#(height, width)=rescaled_frame.shape[:2]
