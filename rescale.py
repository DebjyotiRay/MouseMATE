import cv2 as cv


def rescaleFrame(frame, scale):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)

    dimensions=(width,height)
    return  cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)

def play():
    video = cv.VideoCapture('mp_env\dog2.mp4')
    while True:
        isTrue, frame=video.read()
        rescaled_frame=rescaleFrame(frame,0.5)
        cv.imshow('Frame rescaled', rescaled_frame)
        stop()
def stop():
    if cv.waitKey(1) & 0xFF == ord('F'):
        exit()

