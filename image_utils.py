import numpy as np
import math
import cv2
import os
import optical_flow

def get_frames(filename):
    dirname = os.path.join(os.path.dirname(__file__), 'data\\orig')
    filename = os.path.join(dirname, filename)
    print(filename)
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(dirname, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def read_frames(names, num):
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
    for i in range(np.shape(frames)[0]):
        path = 'data//animated//frame%d.jpg' % i
        #path = 'data//orig//anim_frame%d_1.jpg' % (i*3)
        print("\t"+ path)
        cv2.imwrite(path, frames[i])

def frames_to_video(frames):
    video = cv2.VideoWriter('data//animated//video.avi', 0, 1, np.shape(frames[0]), 0)
    for frame in frames:
        vidout=cv2.resize(frame, np.shape(frames[0]))
        video.write(np.uint8(vidout))
    
    cv2.destroyAllWindows()
    video.release()

def threshold_image(images, levels):
    thresholded = np.zeros(np.shape(images))
    for i in range(len(images)):
        im = images[i]
        rows,cols = np.shape(im)
        for r in range(rows):
            for c in range(cols):
                im[r,c] = levels[np.abs(levels - im[r,c]).argmin()]
        thresholded[i,:,:] = im
    return thresholded

def edge_detector(images,thresh,mval):
    image_edges = np.zeros(np.shape(images))
    h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i,im in enumerate(images):
        horiz_edges = cv2.filter2D(im, -1, h)
        vert_edges = cv2.filter2D(im, -1, h.T)
        #vert_edges = np.zeros(np.shape(horiz_edges))
        #horiz_edges = np.zeros(np.shape(vert_edges))
        edges = np.sqrt(np.square(horiz_edges) + np.square(vert_edges))
        if not thresh == None:
            edges[edges < thresh] = 0
            #edges[edges > thresh] = mval
        image_edges[i,:,:] = edges
    return image_edges