#!/usr/bin/env python3
"""
Matplotlib-animated optical flow
Adapted from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

The original OpenCV code is significantly faster than the introduced
Matplotlib here, so why should we use it? In this case, we probably shouldn't,
but this example gives (I think) a nice general solution for combining a plot
with a video. 

Virtually all the rendering time comes from the `plot_as_ndarray` function,
which is called twice as part of a hack to easily align the image coordinates.
But this code could easily be adapted to, for example, show the tracking plot
nested in the upper right corner of the video, perhaps with a transparent
background, etc, making it a flexible base for incorporating matplotlib graphs
with video.
"""
import numpy as np
import cv2 as cv
from PIL import Image

import matplotlib.pyplot as plt
import io
from time import time

def get_paths(fname):
    cap = cv.VideoCapture(fname)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    _got_frame, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    lines = []
    while True:
        got_frame, frame = cap.read()
        if not got_frame:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        curr_lines = []
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # confusing formatting for compatibility: 
                # a,c are the x-coordinates of (old,new), and (b,d) are
                # y-coordinates in an inverted system
                curr_lines.append(([a,c], [frame.shape[0]-b,frame.shape[0]-d]))
                # print(curr_lines[-1])
            p0 = good_new.reshape(-1, 1, 2)
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        lines.append(curr_lines)
    return lines

def plot_as_ndarray(fig, DPI=128):
    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='raw', dpi=DPI, bbox_inches=0)
    plt_width = fig.get_figwidth()*DPI
    plt_height = fig.get_figheight()*DPI
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(plt_height), int(plt_width), -1))
    io_buf.close()
    return img_arr

def gen_frames(in_path: str, DPI=128):
    lines = get_paths(in_path)
    cap = cv.VideoCapture(in_path)
    _, first_frame = cap.read()
    empty_im = np.full_like(first_frame, 255)

    fig = plt.figure()
    ax = fig.add_subplot()
    
    frame_num = 0
    img_mask = None
    while True:
        t_start = time()
        got_frame, frame = cap.read()
        if not got_frame:
            break
        curr_lines = lines[frame_num]
        
        # perform plotting
        plt.cla()
        ax.set(xlim=(0,first_frame.shape[1]))
        ax.set(ylim=(0,first_frame.shape[0]))
        # by turning off axes and setting this axis to cover the whole plot,
        # we can align the matplotlib plot with the image without any calls
        # to ax.imshow(), which makes performance ~5x better
        ax.set_axis_off()
        ax.set_position((0, 0, 1, 1))

        for line in curr_lines:
            # print(line[1])
            ax.plot(line[0], line[1], color='b')
        
        # read plot into numpy array
        plt_mask = plot_as_ndarray(fig, DPI)[:,:,:3]
        if img_mask is None:
            img_mask = plt_mask
        img_mask = np.where(plt_mask < 255, plt_mask, img_mask)

        frame = cv.resize(frame, (img_mask.shape[1], img_mask.shape[0]))
        disp_frame = np.where(img_mask < 255, img_mask, frame)
        yield disp_frame
        
        print("Rendered next frame: %0.3f sec" % (time() - t_start))
        print("Frame dims: ", frame.shape[:2])
        frame_num += 1
    

if __name__ == "__main__":
    in_path = 'in/traffic.mp4'

    # for frame in gen_frames(in_path):
    #     cv.imshow('frame', frame)
    #     k = cv.waitKey(30) & 0xff
    # cv.destroyAllWindows()

    pil_frames = [Image.fromarray(x[:,:,[2,1,0]], "RGB") for x in gen_frames(in_path)]
    pil_frames[0].save('out/optical_flow.gif', save_all=True, 
        append_images=pil_frames)        
