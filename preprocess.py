import json
import os

_PATH_ = "C:/Users/elena/Desktop/WeHigher/model/labelled/" # 폴더 경로

file_name = []
file_name = os.listdir(_PATH_) # 해당 경로에서 파일 이름들 가져오기

X = []
Y = []
frames = []

for file in file_name:
    file_path = _PATH_ + file
    
    with open(file_path, 'rt', encoding='UTF-8') as f:
        data = json.load(f)
        korean_text = data['korean_text']
        left_hand = data['left_hand']
        right_hand = data['right_hand']
        frame = data['time']

        keypoints = []
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
                    if left_confidence > 0.6 and right_confidence > 0.6:
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
                        keypoints.append(pair)
        
        print(keypoints)

        # keypoints = []
        # keypoints.append(left_hand)
        # keypoints.append(right_hand)

        # X.append(keypoints)
        # Y.append(korean_text)
        # frames.append(frame)