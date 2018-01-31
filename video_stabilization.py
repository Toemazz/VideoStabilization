# http://nghiaho.com/?p=2208
import cv2
import numpy as np
import pandas as pd
import tqdm
import imutils
import os


class VideoStabilizer:
    def __init__(self, video_in_path, video_out_path, side_by_side=False, crop_percent=None, max_width=500):
        # Initialize arguments
        self.video_in_path = video_in_path
        self.video_out_path = video_out_path
        self.side_by_side = side_by_side
        self.crop_percent = crop_percent
        self.max_width = max_width

        # Set up video capture
        self.video = cv2.VideoCapture(self.video_in_path)
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

        # Read 'prev', convert to greyscale and get dimensions
        _, self.prev = self.video.read()
        self.prev_g = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
        (self.frame_h, self.frame_w) = self.prev_g.shape
        print('[INFO]: Setup completed')

        # Calculate trajectory data
        self.calculate_trajectory_data()

    def calculate_trajectory_data(self):
        print('[INFO]: Trajectory data calculations starting....')
        transform_data = []

        for _ in tqdm.tqdm(np.arange(self.n_frames)):
            # Read frame
            ok, curr = self.video.read()

            if ok:
                # Convert to greyscale
                curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

                # Keypoint detection in 'prev_g'
                prev_pts = cv2.goodFeaturesToTrack(self.prev_g, maxCorners=200, qualityLevel=0.01, minDistance=30.0,
                                                   blockSize=3)

                # Calculate optical flow (Lucas-Kanade Method)
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_g, curr_g, prev_pts, None)
                curr_kpts, prev_kpts = [], []

                # Save common keypoints found in 'prev' and 'curr'
                for j, stat in enumerate(status):
                    if stat == 1:
                        # Save keypoints that appear in both
                        prev_kpts.append(prev_pts[j])
                        curr_kpts.append(curr_pts[j])

                # Estimate partial transform
                curr_kpts, prev_kpts = np.array(curr_kpts), np.array(prev_kpts)
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
                self.prev = curr[:]
                self.prev_g = curr_g[:]

        # Calculate the cumulative sum of all transforms for the trajectory
        transform_data = np.array(transform_data)
        trajectory = np.cumsum(transform_data, axis=0)

        # Calculate rolling mean to smooth trajectory, 'backfill' and save to CSV file
        trajectory = pd.DataFrame(trajectory)
        smoothed_trajectory = trajectory.rolling(window=20, center=False).mean()
        smoothed_trajectory = smoothed_trajectory.fillna(method='bfill')

        # Remove 'trajectory', replace with 'smoothed_trajectory' and save to CSV file
        new_transform_data = transform_data + (smoothed_trajectory - trajectory)
        print('[INFO]: Trajectory data calculations finished')

        # Start actual video stabilization
        self.video_stabilization(new_transform_data)

    def video_stabilization(self, transform_data):
        print('[INFO]: Actual video stabilization starting....')
        # Initialize transformation matrix
        t = np.zeros((2, 3))
        transform_data = np.array(transform_data)

        # Setup video capture
        cap = cv2.VideoCapture(self.video_in_path)

        # Set output width and resize
        w_write = min(self.frame_w, self.max_width)
        h_write = imutils.resize(self.prev_g, width=w_write).shape[0]

        # Double output width if 'side_by_side' is 'True'
        if self.side_by_side:
            w_write = w_write * 2

        # Setup video writer
        video_out = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc('P', 'I', 'M', '1'),
                                    self.fps, (w_write, h_write), True)

        for k in np.arange(self.n_frames-1):
            # Read frame
            _, curr = cap.read()

            # Read/build transformation matrix
            t[0, 0] = np.cos(transform_data[k][2])
            t[0, 1] = -np.sin(transform_data[k][2])
            t[0, 2] = transform_data[k][0]
            t[1, 0] = np.sin(transform_data[k][2])
            t[1, 1] = np.cos(transform_data[k][2])
            t[1, 2] = transform_data[k][1]

            # Apply saved transform
            curr_t = cv2.warpAffine(curr, t, (self.frame_w, self.frame_h))

            # Crop current frame with transform applied
            curr_t = self.border_crop(curr_t, crop_percent=self.crop_percent)

            if self.side_by_side:
                # Also crop current frame without transform applied
                curr = self.border_crop(curr, crop_percent=self.crop_percent)

                # Resize to 'max_width' if 'frame_w' > than 'max_width'
                curr = imutils.resize(curr, width=min(self.frame_w, self.max_width))
                curr_t = imutils.resize(curr_t, width=min(self.frame_w, self.max_width))

                # Stack horizontally
                frame_out = np.hstack((curr, curr_t))
            else:
                # Resize to 'max_width' if 'frame_w' > than 'max_width'
                frame_out = imutils.resize(curr_t, width=min(self.frame_w, self.max_width))

            # Display frame
            cv2.imshow('Output', frame_out)
            cv2.waitKey(20)

            # Write frame to output video
            video_out.write(frame_out)

        print('[INFO]: Actual video stabilization finished')
        print('[INFO]: {} saved in {}'.format(self.video_out_path.split('/')[-1],
                                              os.path.dirname(self.video_out_path)))

    @staticmethod
    def border_crop(frame, crop_percent):

        if crop_percent is None:
            return frame

        crop_percent = crop_percent / 100

        if crop_percent >= 50:
            print('[ERR]: You cant crop the whole image!')

        if frame.shape[-1] > 1:
            h, w, _ = frame.shape
        else:
            h, w = frame.shape

        h_crop, w_crop = int(h * crop_percent), int(w * crop_percent)

        return frame[h_crop:h - h_crop, w_crop:w - w_crop]


# Example call to 'VideoStabilizer'
VideoStabilizer('videos/video1.mp4', 'output/video1_out_crop.avi', side_by_side=True, crop_percent=None)
