from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from dotenv import load_dotenv
import os
import openai
import numpy as np

app = Flask(__name__)
CORS(app)
load_dotenv('.env.local')
API_KEY = os.getenv('API_KEY')
openai.api_key = API_KEY

@app.route('/sign', methods=['POST'])
def wordRecognition():

    data = request.get_json()  # JSON 데이터를 받아옴
    # print(data)
    # data 전처리
    empty = np.zeros((78,))
    full_data = np.empty(shape=(0, 2, 78))
    left_data = []
    right_data = []

    for d in data:
        # 양손 일 때
        if len(d['left']) != 0 and len(d['right']) != 0:
            left_data = np.array(d['left'], dtype=np.float32)
            right_data = np.array(d['right'], dtype=np.float32)
        elif len(d['left']) != 0 and len(d['right']) == 0:
            left_data = np.array(d['left'], dtype=np.float32)
            right_data = empty
        elif len(d['left']) == 0 and len(d['right']) != 0:
            left_data = empty
            right_data = np.array(d['right'], dtype=np.float32)
        pair = np.array([left_data, right_data])
        full_data = np.vstack((full_data, [pair])) # shape = (30, 2, 78) => (1, 199, 156)



    # 모델 파일(.h5)을 불러오기
    model = load_model('model4.h5')
    labels = np.load('classes4.npy')
    full_data = full_data.reshape(full_data.shape[0], -1) # (30, 156)
    tmp = np.zeros(shape=(199, 156))
    tmp[:full_data.shape[0], :] = full_data # (199, 156)

    data = np.empty(shape=(0, 199, 156))
    data = np.vstack((data, [tmp])) # (1, 199, 156)

    predictions = model.predict(data)
    word = labels[predictions[0].argmax()]
    print(word)
    return jsonify(word)


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
    print(response["choices"][0]["message"]["content"])
    return jsonify(response["choices"][0]["message"]["content"])
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)

