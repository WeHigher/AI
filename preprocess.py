import json
import os
import numpy as np
from tqdm import tqdm

_PATH_ = "C:/Users/elena/Desktop/WeHigher/model/labelled/" # 폴더 경로

file_name = []
dir_name = os.listdir(_PATH_) # 해당 경로에서 폴더 이름들 가져오기

X = np.empty(shape=(0, 974, 21, 2, 2)) # 총 n개의 단어에는 각각 974 프레임, 각 프레임에는 2개의 keypoints(왼손, 오른손), 각 keypoint에는 2개의 좌표(x, y)
Y = []

for dir in tqdm(dir_name, desc='folder', position=0):
    PATH_DIR = _PATH_ + dir # json 데이터 있는 폴더 
    file_name = os.listdir(PATH_DIR) # 폴더 내 json 파일 이름들 가져오기
    for file in tqdm(file_name, desc='file', position=1, leave=False):
        file_path = PATH_DIR + '/' + file
    
        with open(file_path, 'rt', encoding='UTF-8') as f: # json 파일 open
            data = json.load(f)
            korean_text = data['korean_text']
            left_hand = data['left_hand']
            right_hand = data['right_hand']
            frame = data['time']

            keypoints = np.empty(shape=(0, 21, 2, 2))
            input_keypoints = np.zeros(shape=(974, 21, 2, 2)) # [0, ]
            for i in range(len(left_hand)): # 프레임마다 접근
                tmp = np.empty(shape=(0, 2, 2))
                for j in range(21):
                    pair = np.empty(shape=(2, 2))
                    left_x = 0
                    left_y = 0
                    right_x = 0
                    right_y = 0
                    left = np.empty(shape=2)
                    right = np.empty(shape=2)
                    
                    left_x = left_hand[i][j*3]
                    left_y = left_hand[i][j*3+1]
                    left = np.array([left_x, left_y])

                    right_x = right_hand[i][j*3]
                    right_y = right_hand[i][j*3+1]
                    right = np.array([right_x, right_y])

                    pair = np.array([left, right])
                    tmp = np.vstack((tmp, [pair]))

                keypoints = np.vstack((keypoints, [tmp]))

            input_keypoints[:keypoints.shape[0]] = keypoints

        X = np.vstack((X, [input_keypoints]))
        Y.append(korean_text)

Y = np.array(Y)