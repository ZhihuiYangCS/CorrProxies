import os
import random
import time
from typing import List

import cv2
import rootpath

rootpath.append()
from paths import UCF101_DATA_DIR


def extract_images(video_input_file_path, image_output_dir_path) -> List[str]:
    """
    extract frame images from a video
    :param video_input_file_path: video path
    :param image_output_dir_path: a directory to write all frame images
    :return: a list of frame paths
    """
    count = 0
    # path = os.path.join(os.path.split(os.path.split(image_output_dir_path)[0])[1],
    #                     os.path.split(image_output_dir_path)[1])
    output_frame_paths = []
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            frame_path = os.path.join(image_output_dir_path, "frame_%d.jpg" % count)
            cv2.imwrite(frame_path, image)  # save frame as JPEG file
            # output_frame_paths.append(os.path.join(path, "frame_%d.jpg" % count))
            output_frame_paths.append("frame_%d.jpg" % count)
            count = count + 1
    return output_frame_paths


def ucf101_shuffle_split():
    """
    read all video paths, extract frame images from video
    """
    src_dir = os.path.join(UCF101_DATA_DIR, 'UCF-101')
    file_names = []
    for file_1 in os.listdir(src_dir):
        file_1_path = os.path.join(src_dir, file_1)
        for file_2 in os.listdir(file_1_path):
            file_names.append(os.path.join(file_1, file_2))

    frames_path = os.path.join(UCF101_DATA_DIR, 'frames')
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    file_name_frames_dict = {}
    for file_name in file_names:
        frame_path_i = frames_path
        path_splits = os.path.split(file_name)
        for src_path in path_splits[:-1]:
            frame_path_i = os.path.join(frame_path_i, src_path)
        frame_path_i = os.path.join(frame_path_i, path_splits[-1].split(".")[0])
        if not os.path.exists(frame_path_i):
            os.makedirs(frame_path_i, exist_ok=True)
        frame_paths = extract_images(os.path.join(src_dir, file_name), frame_path_i)
        file_name_frames_dict[file_name] = frame_paths
    file_names += file_names
    random.seed(1)
    random.shuffle(file_names)
    print(len(file_names))


if __name__ == '__main__':
    time1 = time.time()
    ucf101_shuffle_split()
    time2 = time.time()
    print("time = " + str(time2 - time1))
