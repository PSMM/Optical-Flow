#
# block_matching_flow.py
#
# Written by : Pascal Mettes.
#
# The code below provides an implementation of the block matching algorithm
# using SSD to compute the distances between local blocks. Feel free to use it,
# but the code may contain bugs or inaccuracies.
#

import cv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

#
# Compute the Sum of Squared Distances between two equally sized vectors.
#
# Input : The two vectors (both numpy arrays).
# Output: SSD (float).
#
def ssd(arr1, arr2):
    assert len(arr1) == len(arr2)
    return sum((arr1 - arr2) ** 2)

#
# Perform block matching optical flow for two consecutive frames.
#
# Input : The two frames (both numpy arrays), the window size, the area around
#         the window in the previous frame, and the stride between two
#         estimations (all integers).
# Output: The x- and y-velocities (both numpy arrays).
#
def block_matching(im1, im2, window_size, shift, stride):
    # Initialize the matrices.
    vx = np.zeros(((im2.shape[0] - window_size)/float(stride)+1, \
            (im2.shape[1] - window_size)/float(stride)+1))
    vy = np.zeros(((im2.shape[0] - window_size)/float(stride)+1, \
            (im2.shape[1] - window_size)/float(stride)+1))
    wh = window_size / 2
    
    # Go through all the blocks.
    tx, ty = 0, 0
    for x in xrange(wh, im2.shape[0] - wh - 1, stride):
        for y in xrange(wh, im2.shape[1] - wh - 1, stride):
            nm = im2[x-wh:x+wh+1, y-wh:y+wh+1].flatten()
            
            min_dist = None
            flox, flowy = 0, 0
            # Compare each block of the next frame to each block from a greater
            # region with the same center in the previous frame.
            for i in xrange(max(x - shift, wh), min(x + shift + 1, im1.shape[0] - wh - 1)):
                for j in xrange(max(y - shift, wh), min(y + shift + 1, im1.shape[1] - wh - 1)):
                    om = im1[i-wh:i+wh+1, j-wh:j+wh+1].flatten()
                    
                    # Compute the distance and update minimum.
                    dist = ssd(nm, om)
                    if not min_dist or dist < min_dist:
                        min_dist = dist
                        flowx, flowy = x - i, y - j
            
            # Update the flow field. Note the negative tx and the reversal of
            # flowx and flowy. This is done to provide proper quiver plots, but
            # should be reconsidered when using it.
            vx[-tx,ty] = flowy
            vy[-tx,ty] = flowx
            
            ty += 1
        tx += 1
        ty = 0
    
    return vx, vy

#
# Show the quiver plot of the 2D velocities.
#
# Input : The x- and y-velocities (both 2d numpy arrays).
# Output: The quiver plot (matplotlib figure).
#
def quiver_flow_field(vx, vy):
    # Generate the quiver plot.
    plt.figure()
    plt.quiver(vx,vy,color='r')
    plt.show()

#
# Main function of the script. In here, the video capture is loaded, the user
# parameters are yielded, and the block matching flow is called for consecutive
# frames.
#
# Input : -
# Output: -
#
def main():
    assert len(sys.argv) == 6

    # Load the provided capture and yield use parameters.
    try:
        capture     = cv.CaptureFromFile(sys.argv[1])
        nr_frames   = int(sys.argv[2])
        window_size = int(sys.argv[3])
        shift       = int(sys.argv[4])
        stride      = int(sys.argv[5])
    except:
        print "Incorrect parameter(s) provided."
        print "Usage:", sys.argv[0], "#frames window_size stride"
        return
    
    # Assert that both the window size and spatial shift are odd numbers.
    assert window_size % 2 == 1 and shift % 2 == 1
    
    # Retrieve the frames, convert to grayscale, and add to the list.
    frames = []
    for i in xrange(nr_frames):
        frame  = cv.QueryFrame(capture)
        gframe = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)
        cv.CvtColor(frame, gframe, cv.CV_BGR2GRAY)
        frames.append(np.asarray(gframe))
    
    # Compute the flow fields for the frames.
    for i in xrange(1, len(frames)):
        vx, vy = block_matching(frames[i-1], frames[i], window_size, shift, stride)
        quiver_flow_field(vx, vy)
#
# Primary entry point of the script when run directly.
#
if __name__ == "__main__":
    main()
