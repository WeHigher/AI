import json
import os
from tqdm import tqdm
import numpy as np

_PATH_ = "/content/drive/MyDrive/labelled/" # 최초 경로

file_name = []
dir_name = os.listdir(_PATH_) # 해당 경로에서 폴더 이름들 가져오기

X = []
Y = []
frames = []

for dir in tqdm(dir_name):
    PATH_DIR = _PATH_ + dir # json 데이터 있는 폴더 
    file_name = os.listdir(PATH_DIR) # 폴더 내 json 파일 이름들 가져오기
    for file in tqdm(file_name):
        file_path = PATH_DIR + '/' + file
        
        with open(file_path, 'rt', encoding='UTF-8') as f: # json 파일 open
            data = json.load(f)
            korean_text = data['korean_text']
            left_hand = data['left_hand']
            right_hand = data['right_hand']
            frame = data['time']

            keypoints = []
            tmp = []
            for i in range (len(left_hand)):
                for j in range(21 * 3):
                    if j % 3 == 2:
                        left_confidence = left_hand[i][j]
                        right_confidence = right_hand[i][j]

                        pair = []
                        left_x = []
                        left_y = []
                        right_x = []
                        right_y = []
                        left = []
                        right = []
                        if left_confidence > 0 and right_confidence > 0:
                            left_x = left_hand[i][j-2]
                            left_y = left_hand[i][j-1]
                            left.append(left_x)
                            left.append(left_y)

                            right_x = right_hand[i][j-2]
                            right_y = right_hand[i][j-1]
                            right.append(right_x)
                            right.append(right_y)

                            pair.append(left)
                            pair.append(right)
                            tmp.append(pair)

                keypoints.append(tmp)    
            X.append(keypoints)
            Y.append(korean_text)
            frames.append(frame)

np.save(_PATH_ + 'X.npy', X)
np.save(_PATH_ + 'Y.npy', Y)
np.save(_PATH_ + 'frames.npy', frames)