import os
import cv2
import joblib
import pickle
import numpy as np
from ultralytics import YOLO
from collections import deque
# from bytetracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from helper import *
from feature_calculation import *

MIN_THRESHOLD = 0.5
LEN_HISTORY = 15

class ByteTrackArgument:
    track_thresh = 0.5
    track_buffer = 50
    match_thresh = 0.8
    aspect_ratio_thresh = 10.0
    min_box_area = 1
    mot20 = False

def drone_tracking(INPUT_VIDEO_PATH, save_result=True):
    model = YOLO('weights/best.pt')

    trajectory_model = joblib.load('weights\GB_model_15.pkl')
    # with open('weights\GB_model_15.pkl', 'rb') as f:
    #     trajectory_model = pickle.load(f)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_result:
        save_folder = 'output_videos'
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, INPUT_VIDEO_PATH.split("/")[-1][:-4] + ".avi")
        print(f"video save_path is {save_path}")
        txt_path = os.path.join(save_folder, INPUT_VIDEO_PATH.split("/")[-1][:-4] + ".txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    tracker = BYTETracker(ByteTrackArgument) #BYTETracker(track_thresh=0.5, track_buffer=50, match_thresh=0.8)
    frame_id = 0
    history = deque()

    while True:
        print(f'{frame_id} from {all_frames}')
        ret_val, online_im = cap.read()
        if not ret_val:
            break
        outputs = model.predict(source=online_im, conf=MIN_THRESHOLD, verbose=False)
        img_height, img_width = outputs[0].boxes.orig_shape
        outputs = outputs[0].boxes.data
        all_tlwhs = []
        all_ids = []
        all_classes = []
        all_scores = []
        class_outputs = outputs[outputs[:, 5] == 0][:,:5]
        if class_outputs.size()[0] > 0:
            online_targets = tracker.update(class_outputs, [img_height, img_width], [img_height, img_width])
            if len(online_targets) == 0:
                frame_id += 1
                continue

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = []

            for t in online_targets:
                # tlwh = t[0:4]
                # tlwh[2] = tlwh[2] - tlwh[0]
                # tlwh[3] = tlwh[3] - tlwh[1]
                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)  #online_ids.append(t[4])
                    online_classes.append(0)  #online_classes.append(int(t[5]))
                    online_scores.append(score)  #online_scores.append(t[6])

                    # if save_result:
                    #     f = open(txt_path, "a")
                    #     f.write(f"{frame_id},{t[4]},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t[6]:.2f},-1,-1,-1\n")
                    #     f.close()
                    
            all_tlwhs += online_tlwhs
            all_ids += online_ids
            all_classes += online_classes
            all_scores += online_scores

        if len(history) < LEN_HISTORY:
            history.append((all_ids, all_tlwhs, all_classes, all_scores))
        else:
            history.popleft()
            history.append((all_ids, all_tlwhs, all_classes, all_scores))

        if save_result and len(all_tlwhs) > 0:
            online_im = plot_tracking(online_im, history, trajectory_model)
            online_im = cv2.resize(online_im, (int(width), int(height)))
            vid_writer.write(online_im)

        frame_id += 1
    
    
if __name__ == "__main__":
    drone_tracking('video09.avi')
    # PATH = 'videos'
    # for video_name in ['V_DRONE_110.mp4']:
    #     print(f'Video {video_name} is proccesing\n')
    #     video_path = os.path.join(PATH, video_name)
    #     drone_tracking(video_path)
