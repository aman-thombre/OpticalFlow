import numpy as np
import math
import cv2

def smooth_video(frames):
    '''
    Apply a gaussian smoothing filter to the frames.
    INPUTS:
        frames - np array of frames of our video.
    OUTPUTS:
        frames_fil - smoothed frames as np array.
    '''
    ksize = 7
    sigma = 1.5
    lpf = cv2.getGaussianKernel(ksize, sigma) # 1D
    lpf = lpf*np.transpose(lpf) # 2D kernel gaussian filter
    frames_fil = np.zeros(np.shape(frames))
    for i in range(np.shape(frames)[0]):
        frames_fil[i,:,:] = cv2.filter2D(frames[i], -1, lpf)
    return frames_fil

def gradients(frames):
    '''
    Calculate gradients in time, x, and y directions.
    INPUTS:
        frames - smoothed frames of our video.
    OUTPUTS:
        gt - gradient in time axis
        gr - gradient along rows (vertical gradient)
        gc - gradient along cols (horizontal gradient)
    '''
    h = np.array([0.5, 0, -0.5]) # central finite difference filter
    T, R, C = np.shape(frames) # grab num frames, rows, columns
    
    gt = np.zeros((T,R,C)) # time gradient
    gr = np.zeros((T,R,C)) # row gradient (vertical)
    gc = np.zeros((T,R,C)) # col gradient (horizontal)
    
    # time gradient, convolve along time axis
    for r in range(R):
        for c in range(C):
            gt[:,r,c] = np.convolve(h, frames[:,r,c], mode='same')

    # for row (x) gradient, convolve along row axis    
    for t in range(T):
        for c in range(C):
            gr[t,:,c] = np.convolve(h, frames[t,:,c], mode='same')

    # col (x) gradient, convolve along col axis
    for t in range(T):
        for r in range(R):
            gc[t,r,:] = np.convolve(h, frames[t,r,:], mode='same')
    
    # set gradients at edges to 0
    gt[0,:,:]=0
    gt[-1,:,:]=0 
    
    gr[:,0,:]=0
    gr[:,-1,:]=0
    
    gc[:,:,0]=0
    gc[:,:,-1]=0
    
    return gt, gr, gc

def lucas_kanade(gt, gr, gc, H, W):
    '''
    Performs Lucas-Kanade algorithm given gradients.
    INPUTS:
        gt - time gradient
        gr - vertical gradient
        gc - horizontal gradient
        H - height of optical flow block
        W - width of optical flow block
    OUTPUTS:
        vr - vertical velocity
        vc - horizontal velocity
    '''
    T,R,C = np.shape(gt) # set up arrays for vertical, horizontal vels for each frame
    vr = np.zeros((T, int(R/H), int(C/W)))
    vc = np.zeros((T, int(R/H), int(C/W)))
    
    for frame in range(T): # for each frame
        rcount = 0
        for r in range(0,R,H):
            ccount = 0
            for c in range(0,C,W):
                b = (gt[frame,r:r+W,c:c+H]).flatten(order='C') # set up b vector (time)
                a1 = (gc[frame,r:r+W,c:c+H]).flatten(order='C') # set up first col of a matrix (x grad)
                a2 = (gr[frame,r:r+W,c:c+H]).flatten(order='C') # second col a matrix (y grad)
                
                b = np.transpose(b) * -1
                A = np.column_stack((a1, a2)) # a nd b matrices
                
                v = np.dot(np.linalg.pinv(A),b) # solve Av = b
                
                if rcount < int(R/H) and ccount < int(C/W):
                    vr[frame,rcount,ccount] = v[0] # set the velocities
                    vc[frame,rcount,ccount] = v[1]
                
                ccount+=1
            
            rcount+=1
    
    return vc, vr

def medianfilt(x, H, W):
    '''
    Median-filter the video of pixel velocities.
    INPUTS:
        x - pixel velocities video
        H - the height of median-filtering blocks
        C - the width of median-filtering blocks
    OUTPUTS:
        y - median-filtered x, quantized
    '''
    T,R,C = np.shape(x)
    y = np.zeros((T,R,C))
    
    for t in range(T):
        for r in range(R):
            for c in range(C):
                rmin = max(0,r-int((H-1)/2)) # row and column range
                rmax = min(R,r+int((H-1)/2)+1)
                cmin = max(0,c-int((W-1)/2))
                cmax = min(C,c+int((W-1)/2)+1)
                y[t,r,c] = np.median(x[t,rmin:rmax,cmin:cmax]) # calc median
                
    return y # return  median velocities

def interpolate(x, U):
    '''
    Upsample and interpolate an image using bilinear interpolation.

    x (TxRxC) - a video with T frames, R rows, C columns
    U (scalar) - upsampling factor
    y (Tx(U*R)x(U*C)) - interpolated image
    '''
    
    #using interp
    
    T,R,C = np.shape(x)
    
    x1 = np.zeros((T, R, U*C))
    y = np.zeros((T, U*R, U*C))
    
    for t in range(T):
        for r in range(R):
            x1[t,r,:] = np.interp(np.arange(U*C), U*np.arange(C), x[t,r,:])
        
        for c in range(C*U):
            y[t,:,c] = np.interp(np.arange(U*R), U*np.arange(R), x1[t,:,c])
        
    return np.rint(y)
    
    
    '''
    #upsample, then first order hold
    T,R,C = np.shape(x)
    
    x1 = np.zeros((T,R,C*U))
    y = np.zeros((T,R*U,C*U))
    h = np.zeros((2*U+1))
    
    for i in range(U):
        h[i] = i/U
    
    for i in range(U, 2*U+1):
        h[i] = 1 - (i-U)/U
    
    for t in range(T):
        for r in range(R):
            for c in range(C*U):
                if c % U == 0:
                    x1[t,r,c] = x[t,r,int(c/U)]
    
        for c in range(C*U):
            for r in range(R*U):
                if r % U == 0:
                    y[t,r,c] = x1[t,int(r/U),c]
                    
        for r in range(R*U):
            y[t,r,:] = np.convolve(h, y[t,r,:], 'same')
        
        for c in range(C*U):
            y[t,:,c] = np.convolve(h, y[t,:,c], 'same')
                    
    return y
    '''

def velocity_fill(x, vr, vc, keep):
    '''
    Fill in missing frames by copying samples with a shift given by the velocity vector.
    INPUTS:
        x - video signal we want to apply optical flow to.
        vr - row (vertical) velocity
        vc - horizontal (col) velocity
        keep - list of frames to keep
    OUPUTS:
        
    '''
    T,R,C = np.shape(x)
    Ra = np.shape(vr)[1]
    Cb = np.shape(vr)[2]
    
    y = np.zeros((T,R,C))
    
    for t in range(T):
        if(np.isin(t, keep)):
            y[t,:,:] = x[t,:,:]
            
        else:
            for r in range(R):
                for c in range(C):
                    if r >= Ra or c >= Cb:
                        y[t,r,c] = y[t-1,r,c]
                    else:
                        rcoord = int(max(min(r - vr[t-1,r,c],R-1),0))
                        ccoord = int(max(min(c - vc[t-1,r,c],C-1),0))
                        y[t,r,c] = y[t-1, rcoord, ccoord]
                        
    return y