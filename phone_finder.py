import numpy as np
import cv2
from skimage import data, filters

# # Open Video
# # cap = cv2.VideoCapture('./data/drones/videos/V_DRONE_107.mp4')
# cap = cv2.VideoCapture('video17.avi')

# # Randomly select 25 frames
# frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# # Store selected frames in an array
# frames = []
# for fid in frameIds:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
#     ret, frame = cap.read()
#     frames.append(frame)

# # Calculate the median along the time axis
# medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# # Reset frame number to 0
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# # Convert background to grayscale
# grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# # Loop over all frames
# ret = True
# while(ret):

#   # Read frame
#   ret, frame = cap.read()
#   # Convert current frame to grayscale
#   gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   # Calculate absolute difference of current frame and 
#   # the median frame
#   dframe = cv2.absdiff(gframe, grayMedianFrame)
#   # Treshold to binarize
#   th, dframe = cv2.threshold(dframe, 15, 255, cv2.THRESH_BINARY)
#   frame = cv2.bitwise_and(frame, frame, mask=dframe)
#   # Display image
#   cv2.imshow('frame', frame)
#   cv2.waitKey(20)

# # Release video object
# cap.release()

# # Destroy all windows
# cv2.destroyAllWindows()


def phone_finder(video_path):
  cap = cv2.VideoCapture(video_path)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  result_path = 'res_' + video_path
  vid_writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

  frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
  frames = []
  for fid in frameIds:
      cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
      ret, frame = cap.read()
      frames.append(frame)

  medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
  ret = True
  count = 0
  while(ret):
    count += 1
    print(count)
    ret, frame = cap.read()
    if ret:
      gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      dframe = cv2.absdiff(gframe, grayMedianFrame)
      th, dframe = cv2.threshold(dframe, 15, 255, cv2.THRESH_BINARY)
      frame = cv2.bitwise_and(frame, frame, mask=dframe)
      vid_writer.write(frame)

  return result_path

phone_finder('cam0.mp4')