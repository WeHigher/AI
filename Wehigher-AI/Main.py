from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
import openai

app = Flask(__name__)
CORS(app)
load_dotenv('.env.local')
API_KEY = os.getenv('API_KEY')
openai.api_key = API_KEY

@app.route('/sign', methods=['POST'])
def wordRecognition():

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


@app.route('/sentense', methods=['POST'])
def sentenceCreate():
    data = request.get_json()

    # 모델 - GPT 3.5 Turbo 선택
    model = "gpt-3.5-turbo"

    # 질문 작성하기
    query = '해당 단어를 토대로 문장을 만들어주세요.'
    for word in data:
        query += f"{word}, "

    query = query[:-2]

    # 메시지 설정하기
    messages = [
        {"role": "system", "content": "경어체로 답변하고, 한글로만 답변해주세요."},
        {"role": "user", "content": query}
    ]

    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)

