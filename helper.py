import cv2
import numpy as np

from feature_calculation import *

THRESHOLD_APERANCE = 5
ID2CLASSES = {0: 'drone', 1: 'other'}
TEXT_COLOR = (0, 0, 255)
COLORS = [(0, 255, 0),
          (255, 0, 0)]

text_scale = 1.5
text_thickness = 2
line_thickness = 2

def convert_history_to_dict(track_history):
    history_dict = {}
    for frame_content in track_history:
        obj_ids, tlwhs, _, _ = frame_content
        for obj_id, tlwh in zip(obj_ids, tlwhs):
            tl_x, tl_y, w, h = tlwh
            mid_x, mid_y = int(tl_x + w/2), int(tl_y + h/2)

            if obj_id not in history_dict.keys():
                history_dict[obj_id] = [[mid_x, mid_y]]
            else:
                history_dict[obj_id].append([mid_x, mid_y])

    return history_dict

def plot_tracking(im, track_history, trajectory_model):
    im_h, im_w = im.shape[:2]
    obj_ids, tlwhs, class_ids, scores = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        if len(history_dict[obj_id]) > THRESHOLD_APERANCE:
            history_coords = np.array(history_dict[obj_id])
            x_center_batch = history_coords[:, 0]/im_w
            y_center_batch = history_coords[:, 1]/im_h
            TA = turn_angle(x_center_batch, y_center_batch)
            V = velocity(x_center_batch, y_center_batch)
            C = curvature(x_center_batch, y_center_batch)
            A = acceleration(x_center_batch, y_center_batch)
            CD = CDF(x_center_batch, y_center_batch)
            class_id = trajectory_model.predict([[TA, C, V, A, CD]])[0]
        else:   
            # class_id = class_ids[i]
            continue

        # if class_id==0:
        #     # print(f"{frame_id},{t[4]},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t[6]:.2f},-1,-1,-1\n")
        #     print(f"_,{class_id},{x1},{y1},{w},{h},{scores[i]},-1,-1,-1\n")

        id_text = '{}'.format(int(obj_id))
        if class_id==0:
            color = COLORS[class_id]
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                        thickness=text_thickness)
            cv2.putText(im, ID2CLASSES[class_id], (intbox[0], intbox[3] + 20) , cv2.FONT_HERSHEY_PLAIN, text_scale, color, thickness = text_thickness)

            for idx in range(len(history_dict[obj_id]) - 1):
                prev_point, next_point = history_dict[obj_id][idx], history_dict[obj_id][idx+ 1]
                cv2.line(im, prev_point, next_point, color, 2)

    return im
