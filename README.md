# VideoStabilization

## Description
Reduces distracting vibrations from videos by smoothing the transition between frames in a video

## How it Works
1. Find the transformation from frame[i-1] to frame[i] using optical flow for all frames in the video. The transformation consists of three parameters:
	- dx (x direction)
	- dy (y direction)
	- da (angle)
2. Accumulate the transformations to get the trajectory for `x`, `y` and `a` for each frame
3. Smooth out the trajectory to ensure the image actually appears 'stabilized'.
4. Create a new transformation such that:
`new_transformation = old_transformation + (smoothed_trajectory - trajectory)`
5. Apply the `new_transformation` to the video


## Software Versions
- python    `3.6.1`
- cv2       `3.3.0`
- numpy     `1.2.1`
- pandas    `0.20.3`
- imtils    `0.4.3`
- tqdm      `4.11.2`
