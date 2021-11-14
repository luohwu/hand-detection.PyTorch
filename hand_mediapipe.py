import cv2
import math
import numpy as np
import os
from opt import *
import pandas as pd
# Read images with OpenCV.
from ast import literal_eval

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# help(mp_hands.Hands)

def detect_hands(row):
    frame=row['frame']
    par_id=row['id'][0:3]
    video_id=row['id'][3:]
    frame_name=f'frame_{str(frame).zfill(10)}.jpg'
    frame_path=os.path.join(args.data_path,'rgb_frames',par_id,video_id,frame_name)
    assert os.path.exists(frame_path),f'file not exists: {frame_path}'
    image=cv2.imread(frame_path)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return  None
    hand_bbox=[]
    image_height, image_width, _ = image.shape
    for rect in results.palm_detections:
        hand_bbodx_relative = rect.location_data.relative_bounding_box
        x0 = round(image_width * (hand_bbodx_relative.xmin))
        y0 = round(image_height * (hand_bbodx_relative.ymin))
        x1 = round(image_width * (hand_bbodx_relative.xmin + hand_bbodx_relative.width))
        y1 = round(image_height * (hand_bbodx_relative.ymin + hand_bbodx_relative.height))
        hand_bbox.append(x0)
        hand_bbox.append(y0)
        hand_bbox.append(x1)
        hand_bbox.append(y1)
    return hand_bbox


# Run MediaPipe Hands.
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7) as hands:

    all_par_video_id=sorted(id)
    # all_par_video_id=['P01P01_01','P01P01_02']
    for par_video_id in all_par_video_id:
        participant_id=par_video_id[0:3]
        video_id=par_video_id[3:]
        # print(video_id)
        annos_file_path=os.path.join(args.data_path,annos_path,f'nao_{participant_id}{video_id}.csv')
        print(video_id)
        annos=pd.read_csv(annos_file_path)
        if annos.shape[0]>0:
            annos=annos.rename(columns={'nao_bbox_resiezed':'nao_bbox_resized'})
            annos['hand_bbox']=annos.apply(detect_hands,axis=1)
            annos.to_csv(annos_file_path,index=False)

    #
    # data_path=args.data_path
    # print(data_path)
    # for image_index in range(1,31000,1):
    #     name=os.path.join(data_path, f'frame_{str(image_index).zfill(10)}.jpg')
    #     image=cv2.imread(name)
    #
    #     # Convert the BGR image to RGB, flip the image around y-axis for correct
    #     # handedness output and process it with MediaPipe Hands.
    #     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #
    #     # Print handedness (left v.s. right hand).
    #     print(f'Handedness of {name}:')
    #     print(results.multi_handedness)
    #
    #     if not results.multi_hand_landmarks:
    #       continue
    #
    #     # Draw hand landmarks of each hand.
    #     print(f'Hand landmarks of {name}:')
    #     image_height, image_width, _ = image.shape
    #     for rect in results.palm_detections:
    #         hand_bbodx_relative=rect.location_data.relative_bounding_box
    #         x0 = round(image_width * (hand_bbodx_relative.xmin))
    #         y0 = round(image_height * (hand_bbodx_relative.ymin))
    #         x1 = round(image_width * (hand_bbodx_relative.xmin+hand_bbodx_relative.width))
    #         y1 = round(image_height * (hand_bbodx_relative.ymin+hand_bbodx_relative.height))
    #         cv2.rectangle(image,(x0,y0),(x1,y1),(0, 0, 255), 3)
    #     cv2.imshow('img',image)
    #     cv2.waitKey(0)

