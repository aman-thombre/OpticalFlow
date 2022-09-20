import numpy as np
import math
import cv2
import os

def get_frames(filename):
    dirname = os.path.join(os.path.dirname(__file__), 'data')
    filename = os.path.join(dirname, filename)
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(dirname, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def read_frames(names, num=98):
    '''
    Read frames into a numpy array.
    INPUTS:
        num - number of frames
    OUTPUTS:
        frames - images in a numpy array, normalized to 1
    '''
    im = cv2.imread(names[0], cv2.IMREAD_GRAYSCALE) 
    shape = (num, np.shape(im)[0], np.shape(im)[1]) # get shape of images
    frames = np.zeros(shape)
    for i in range(len(names)):
        frames[i,:,:] = cv2.imread(names[i], cv2.IMREAD_GRAYSCALE) # put grayscale images into np array
    return frames

def write_frames(frames):
    for i in range(len(frames)):
        path = 'data//animated//frame%d.jpg' % i
        #print(path)
        cv2.imwrite(path, frames[i])

def frames_to_video(frames):
    video = cv2.VideoWriter('data//animated//video.avi', 0, 1, np.shape(frames[0]), 0)
    for frame in frames:
        vidout=cv2.resize(frame, np.shape(frames[0]))
        video.write(np.uint8(vidout))
    
    cv2.destroyAllWindows()
    video.release()
