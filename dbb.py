import os
import cv2
import math
import numpy as np
import pandas as pd
from xml.etree.ElementTree import parse

class VideoReader:
    def __init__(self, video_name, res_video_name):
        self.cap = cv2.VideoCapture(video_name)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_writer = cv2.VideoWriter(res_video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    def get_next_frame(self):
        ret, img = self.cap.read()
        if ret:
            return img
        else:
            raise TypeError("The next frame is not exist")
        
    def write(self, img):
        self.vid_writer.write(img)

    def get_frames_count(self):
        return self.frames

def plot_rectangle(img, xtl, ytl, h, w):
    return cv2.rectangle(img, (xtl, ytl), (xtl+w, ytl+h), color=(0,0,255), thickness=2)

def plot_center(img, xtl, ytl, h, w):
    xc, yc = int(xtl+w/2), int(ytl+h/2)
    return cv2.circle(img, (xc, yc), radius=2, color=(0, 0, 255), thickness=-1)

def bbox_ploter(file_name, rectangle_drawing=True):
    # video_name = 'data/drones/videos/' + file_name + '.mp4'
    # csv_name = 'data/drones/gt/' + file_name + '_LABELS.csv'
    # res_video_name = 'data/drones/output_videos/' + file_name + '_BBOX.mp4'

    res_video_name = file_name + '_BBOX.mp4'
    # if file_name[2:6] == 'BIRD':
    #     path_ = 'data/birds/'
    # else:
    #     path_ = 'data/drones/'
    video_name = file_name + '.avi'#path_ + 'videos/' + file_name + '.mp4'
    # file_name = path_ + 'gt/' + file_name

    video_reader = VideoReader(video_name, res_video_name)
    frames = video_reader.get_frames_count()

    if os.path.exists(file_name + '_LABELS.csv'):
        df = pd.read_csv(file_name + '_LABELS.csv')
        df = df.drop(columns=['Unnamed: 0', 'time'], axis=1)
        for name_column in df.columns:
            df[name_column] = df[name_column].apply(lambda x: x.strip("[]").split(", "))

        if frames != len(df['object_1']):
            raise TypeError("The number of frames not equal number of rows in dataframe")

        for i in range(frames):
            img = video_reader.get_next_frame()
            for name_column in df.columns:
                if not math.isnan(float(df[name_column][i][0])):
                    xtl, ytl, h, w = int(float(df[name_column][i][0])), int(float(df[name_column][i][1])), int(float(df[name_column][i][3])), int(float(df[name_column][i][2]))
                    img = plot_rectangle(img, xtl, ytl, h, w) if rectangle_drawing else plot_center(img, xtl, ytl, h, w)
            video_reader.write(img)

    elif os.path.exists(file_name + '_LABELS.txt'):
        df = np.loadtxt(file_name + '_LABELS.txt', dtype="float", delimiter=",", usecols =(0,1,2,3,4,5))
        
        if frames <= int(max(df[:, 0])):
            raise TypeError("The number of frames not equal number of rows in dataframe")
        
        for i in range(1, frames):
            img = video_reader.get_next_frame()
            for obj in df[df[:, 0] == i, :]:
                xtl, ytl, w, h = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                print(i)
                img = plot_rectangle(img, xtl, ytl, h, w) if rectangle_drawing else plot_center(img, xtl, ytl, h, w)
            video_reader.write(img) 
    else:    
        print(' has not scv or txt file\n')

if __name__ == '__main__':
    bbox_ploter('V_DRONE_Source_3_p22')
    # os.makedirs('data/drones/output_videos', exist_ok = True)
    # for file in os.listdir('data/drones/videos'):
    #     name = file[:-4]
    #     print(name, '\t')
    #     bbox_ploter(name)