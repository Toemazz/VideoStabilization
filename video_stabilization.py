# http://nghiaho.com/?p=2208
import cv2
import numpy as np
import pandas as pd
import tqdm
import imutils


# Required paths
video_path = 'video1.mp4'
output_dir = 'output'
side_by_side = True
max_width = 400

# Set up video capture
video = cv2.VideoCapture(video_path)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
print('[INFO]: Video loaded')

# Read 'prev', convert to greyscale and get dimensions
_, prev = video.read()
prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
(frame_h, frame_w) = prev_g.shape

# -----------------------------------------------------------------
print('[INFO]: Trajectory data calculations starting....')
transform_data = []

for i in tqdm.tqdm(np.arange(n_frames)):
    # Read frame
    ok, curr = video.read()

    if ok:
        curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Keypoint detection in 'prev_g'
        prev_pts = cv2.goodFeaturesToTrack(prev_g, maxCorners=200, qualityLevel=0.01, minDistance=30.0, blockSize=3)

        # Calculate optical flow (Lucas-Kanade Method)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, prev_pts, None)

        curr_kpts, prev_kpts = [], []

        # Save common keypoints found in 'prev' and 'curr'
        for j, stat in enumerate(status):

            if stat == 1:
                # Save keypoints that appear in both
                prev_kpts.append(prev_pts[j])
                curr_kpts.append(curr_pts[j])

        curr_kpts, prev_kpts = np.array(curr_pts), np.array(prev_pts)

        # Estimate partial transform
        transform_new = cv2.estimateRigidTransform(prev_kpts, curr_kpts, False)

        if transform_new is not None:
            transform = transform_new

        # x, y and rotational translations
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])

        # Store for saving to disk as table
        transform_data.append([dx, dy, da])

        # Set current frame to previous frame for the next iteration
        prev = curr[:]
        prev_g = curr_g[:]
# -----------------------------------------------------------------

# -----------------------------------------------------------------
transform_data = np.array(transform_data)

# Calculate the cumulative sum of all transforms for the trajectory
trajectory = np.cumsum(transform_data, axis=0)

# Calculate rolling mean to smooth trajectory, 'backfill' and save to CSV file
trajectory = pd.DataFrame(trajectory)
smoothed_trajectory = trajectory.rolling(window=30, center=False).mean()
smoothed_trajectory = smoothed_trajectory.fillna(method='bfill')
smoothed_trajectory.to_csv('{}/smoothed_trajectory.csv'.format(output_dir))

# Remove 'trajectory', replace with 'smoothed_trajectory' and save to CSV file
new_transform_data = transform_data + (smoothed_trajectory - trajectory)
new_transform_data.to_csv('{}/new_transform_data.csv'.format(output_dir))
print('[INFO]: Trajectory data calculations finished')
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# Applying Video Stabilization
# Initialize transformation matrix
t = np.zeros((2, 3))
new_transform_data = np.array(new_transform_data)

# Setup video capture
cap = cv2.VideoCapture(video_path)

# Set output width and resize
w_write = min(frame_w, max_width)
h_write = imutils.resize(prev_g, width=w_write).shape[0]

# Double output width if 'side_by_side' is 'True'
if side_by_side:
    w_write = w_write * 2

# Setup video writer
video_out = cv2.VideoWriter('{}/stabilized_output_video.avi'.format(output_dir),
                            cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps, (w_write, h_write), True)

for k in np.arange(n_frames-1):
    # Read frame
    _, curr = cap.read()

    # Read/build transformation matrix
    t[0, 0] = np.cos(new_transform_data[k][2])
    t[0, 1] = -np.sin(new_transform_data[k][2])
    t[0, 2] = new_transform_data[k][0]
    t[1, 0] = np.sin(new_transform_data[k][2])
    t[1, 1] = np.cos(new_transform_data[k][2])
    t[1, 2] = new_transform_data[k][1]

    # Apply saved transform
    curr_t = cv2.warpAffine(curr, t, (frame_w, frame_h))

    if side_by_side:
        # Resize to 'max_width' if 'frame_w' > than 'max_width'
        curr = imutils.resize(curr, width=min(frame_w, max_width))
        curr_t = imutils.resize(curr_t, width=min(frame_w, max_width))
        # combine arrays for side by side
        frame_out = np.hstack((curr, curr_t))
    else:
        # Resize to 'max_width' if 'frame_w' > than 'max_width'
        frame_out = imutils.resize(curr_t, width=min(frame_w, max_width))

    # Display frame
    cv2.imshow('Output', frame_out)
    cv2.waitKey(20)

    # Write frame to output video
    video_out.write(frame_out)

print('[INFO]: She done bai')
