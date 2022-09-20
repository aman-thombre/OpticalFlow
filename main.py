import numpy as np
import image_utils
import optical_flow
import matplotlib.pyplot as plt

def main():

    print("Reading frames.")
    frame_names = []
    num_frames=98
    for i in range(num_frames):
        frame_names.append('data\\orig\\frame%d.jpg' % i)
    frames = image_utils.read_frames(frame_names, num_frames) # get frames of video into np array

    keep = [0]
    anim_names = ['data\\orig\\provided\\frame%d.png' % x for x in keep]
    print(anim_names)
    prov_frames = image_utils.read_frames(anim_names, len(anim_names)+1)
    anim_frames = np.zeros(np.shape(frames))
    for i in range(len(keep)):
        anim_frames[keep[i]] = prov_frames[i]

    print('Smoothing video.')
    frames_fil = optical_flow.smooth_video(frames, 7, 1.5) # smooth frames

    print('Calculating gradients.')
    gt, gr, gc = optical_flow.gradients(frames_fil) # gradients of pixels
    
    print("Calculating pixel velocities with Lucas-Kanade.")
    vc, vr = optical_flow.lucas_kanade(gt,gr,gc,4,4) # pixel velocities
    
    print("Starting median filtering.")
    smooth_vc = optical_flow.medianfilt(vc, 3, 3) # median filt horizontal vel
    smooth_vr = optical_flow.medianfilt(vr, 3, 3) # median filt vert vel

    print("Upsampling and interpolating velocities.")
    interp_vc = optical_flow.interpolate(smooth_vc, 4) # interp velocities since lucas-kanade downsamples
    interp_vr = optical_flow.interpolate(smooth_vr, 4)
   
    print("Filling in missing frames.")
    animated = optical_flow.velocity_fill(anim_frames, interp_vc, interp_vr, keep)

    print("Writing frames.")
    image_utils.write_frames(animated)

if __name__ == "__main__":
    main()