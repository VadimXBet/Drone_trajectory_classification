import os
import cv2
import joblib
import pickle
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
# from bytetracker import BYTETracker
import matplotlib.pyplot as plt
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from helper import *
from feature_calculation import *

MIN_THRESHOLD = 0.5
LEN_HISTORY = 15

class inference_detect_model():
    def __init__(self, weights, conf):
        self.model = YOLO(weights)
        self.thres = conf
        self.class_id = 0

    def predict(self, img):
        predictions = self.model.predict(source=img, conf=self.thres, verbose=False)[0]
        predictions = predictions.boxes
        img_height, img_width = predictions.orig_shape
        outputs = predictions.data
        class_outputs = outputs[outputs[:, 5] == self.class_id][:,:5].cpu()
        return img_height, img_width, class_outputs

class Detect_models_pipeline():
    def __init__(self, conf1, conf2):
        self.first_model = inference_detect_model(weights='weights/yolov8m.pt', conf=conf1)
        self.second_model = inference_detect_model(weights='weights\yolov11n_320x320_20.pt', conf=conf1)

    def predict(self, img):
        img_height, img_width, outputs = self.first_model.predict(img)
        if outputs.size()[0] > 0:
            for i, output in enumerate(outputs):
                tlwh = output[0:4]
                old_w, old_h = tlwh[2].item(), tlwh[3].item()
                tlwh[2] = tlwh[2] - tlwh[0]
                tlwh[3] = tlwh[3] - tlwh[1]
                score = output[4]
                x, y, w, h = tlwh
                xc, yc = x + w/2, y + h/2
                x_min = max(0, xc-w)
                x_max = min(img_width, xc+w)
                y_min = max(0, yc-h)
                y_max = min(img_height, yc+h)
                intbox = tuple(map(int, (x_min, y_min, x_max, y_max)))
                crop_img = img[intbox[1]:intbox[3], intbox[0]:intbox[2]]
                _, _, res = self.second_model.predict(crop_img)
                if res.size()[0] > 0:
                    tlwh = res[0][0:4]
                    # tlwh[2] = tlwh[2] - tlwh[0]
                    # tlwh[3] = tlwh[3] - tlwh[1]
                    x, y, _, _ = tlwh
                    x = x + x_min
                    y = y + y_min
                    outputs[i] = torch.tensor([x, y, old_w, old_h, score])
                    # intbox = tuple(map(int, (x, y, x + w, y + h)))
                    # cv2.rectangle(img, intbox[0:2], intbox[2:4], color=(255, 0, 0), thickness=line_thickness)
                else:
                    outputs = torch.cat([outputs[:i], outputs[i+1:]])
                # plt.imshow(img)
                # plt.show()
        return img_height, img_width, outputs

class ByteTrackArgument:
    track_thresh = 0.5
    track_buffer = 50
    match_thresh = 0.8
    aspect_ratio_thresh = 10.0
    min_box_area = 1
    mot20 = False

def drone_tracking(INPUT_VIDEO_PATH, save_result=True):
    # detect_model = inference_detect_model(weights='weights/yolov8m.pt', conf=MIN_THRESHOLD)
    detector_pipline = Detect_models_pipeline(conf1=MIN_THRESHOLD, conf2=0.4)

    # trajectory_model = joblib.load('weights/GB_model_15.pkl')
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
        save_path = os.path.join(save_folder, INPUT_VIDEO_PATH.split("\\")[-1][:-4] + ".avi")
        print(f"video save_path is {save_path}")
        txt_path = os.path.join(save_folder, INPUT_VIDEO_PATH.split("\\")[-1][:-4] + ".txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    tracker = BYTETracker(ByteTrackArgument) #BYTETracker(track_thresh=0.5, track_buffer=50, match_thresh=0.8)
    frame_id = 0
    history = deque()

    while True:
        frame_id += 1
        print(f'{frame_id} from {all_frames}')
        ret_val, online_im = cap.read()
        if not ret_val or frame_id > 100:
            break
        # img_height, img_width, class_outputs = detect_model.predict(online_im)
        img_height, img_width, class_outputs = detector_pipline.predict(online_im)
        all_tlwhs = []
        all_ids = []
        all_classes = []
        all_scores = []
        if class_outputs.size()[0] > 0:
            online_targets = tracker.update(class_outputs, [img_height, img_width], [img_height, img_width])
            if len(online_targets) == 0:
                continue

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = []

            for t in online_targets:
                # tlswh = t[0:4]
                # tlwh[2] = tlwh[2] - tlwh[0]
                # tlwh[3] = tlwh[3] - tlwh[1]

                # score = t[4]
                # tid = 1

                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                print(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n")
                if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)  #online_ids.append(t[4])
                    online_classes.append(0)  #online_classes.append(int(t[5]))
                    online_scores.append(score)  #online_scores.append(t[6])

                    if save_result:
                        f = open(txt_path, "a")
                        f.write(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n")
                        f.close()
                    
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
            online_im = plot_tracking(online_im, history, frame_id, txt_path, INPUT_VIDEO_PATH.split("\\")[-1][:-4])
            online_im = cv2.resize(online_im, (int(width), int(height)))
            vid_writer.write(online_im)
    
if __name__ == "__main__":
    drone_tracking('test_videos\\videos\\video09.avi')
    # PATH = 'test_videos/videos'
    # for video_name in os.listdir(PATH):
    #     print(f'Video {video_name} is proccesing\n')
    #     video_path = os.path.join(PATH, video_name)
    #     drone_tracking(video_path)
