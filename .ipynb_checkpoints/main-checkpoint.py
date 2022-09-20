import numpy as np
import image_utils
import optical_flow
import matplotlib.pyplot as plt

def main():
    frame_names = []
    for i in range(98):
        frame_names.append('data\\orig\\frame%d.jpg' % i)
   # print(frame_names)
    frames = image_utils.read_frames(frame_names) # get frames of video into np array
    #H, W = np.shape(frames[0])

    anim_names = ['data\\orig\\frame0_anim.jpg'] 
    anim_frames = image_utils.read_frames(anim_names)

    frames_fil = optical_flow.smooth_video(frames) # smooth frames
    gt, gr, gc = optical_flow.gradients(frames_fil) # gradients of pixels
    
    
    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(14, 10))
    plt1 = ax1.imshow(gt[55,:,:])
    plt.colorbar(plt1, ax=ax1)
    ax1.set_title('Time Gradient')
    plt2 = ax2.imshow(gr[30,:,:])
    plt.colorbar(plt2, ax=ax2)
    ax2.set_title('Vertical Gradient')
    plt3 = ax3.imshow(gc[20,:,:])
    plt.colorbar(plt3, ax=ax3)
    ax3.set_title('Horizontal Gradient')
    ax4.imshow(frames[10,:,:],cmap='gray')
    plt.show()
    '''
    
    print("Starting Lucas-Kanade.")
    vc, vr = optical_flow.lucas_kanade(gt,gr,gc,4,4) # pixel velocities
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(14, 7))
    plt1 = ax1.imshow(vr[1,:,:])
    plt.colorbar(plt1, ax=ax1)
    ax1.set_title('Vertical Velocity')
    plt2 = ax2.imshow(vc[1,:,:])
    plt.colorbar(plt2, ax=ax2)
    ax2.set_title('Horizontal Velocity')
    plt3 = ax3.imshow(vr[20,:,:])
    plt.colorbar(plt3, ax=ax3)
    ax3.set_title('Vertical Velocity')
    plt4 = ax4.imshow(vc[20,:,:])
    plt.colorbar(plt4, ax=ax4)
    ax4.set_title('Horizontal Velocity')
    plt.show()


    '''print("Starting median filtering.")


    smooth_vc = optical_flow.medianfilt(vc, 2, 2) # median filt horizontal vel
    smooth_vr = optical_flow.medianfilt(vr, 2, 2) # median filt vert vel
    print("Starting velocity fill.")

    
    
    
    animated = optical_flow.velocity_fill(anim_frames, smooth_vc, smooth_vr, [0])
    print("Writing frames.")
    image_utils.write_frames(animated)'''

    #image_utils.frames_to_video(frames)

if __name__ == "__main__":
    main()