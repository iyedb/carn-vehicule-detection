import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from project_functions import *
import tracker

dist_pickle = pickle.load(open("classifier.p", "rb" ))

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]

orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


p = tracker.Tracker(dist_pickle)

stream_output = './project_video_output.mp4'
clip = VideoFileClip('./project_video.mp4')
processed_clip = clip.fl_image(p)
processed_clip.write_videofile(stream_output, audio=False)