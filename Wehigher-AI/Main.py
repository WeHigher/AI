from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

@app.route('/sign', methods=['POST'])
def get_items():

    data = request.get_json()  # JSON 데이터를 받아옴

    # data 전처리
    empty = np.zeros((78,))
    full_data = np.empty(shape=(0, 2, 78))
    left_data = []
    right_data = []

    for left, right in data:
        # 양손 일 때
        if len(left) != 0 and len(right) != 0:
            left_data = np.array([left], dtype=np.float32)
            right_data = np.array([right], dtype=np.float32)
        elif len(left) != 0 and len(right) == 0:
            left_data = np.array([left], dtype=np.float32)
            right_data = empty
        elif len(left) == 0 and len(right) != 0:
            left_data = empty
            right_data = np.array([right], dtype=np.float32)
        pair = np.array([left_data, right_data])
        full_data = np.vstack((full_data, [pair]))

    # 모델 파일(.h5)을 불러오기
    model = load_model('your_model.h5')

    predictions = model.predict(full_data)

    return predictions

if __name__ == '__main__':
    app.run(debug=True)

