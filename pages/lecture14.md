# Lecture 14. Character-level LSTM 실습

이번 강의에서는 Shakespeare 텍스트를 문자 단위로 나누고, LSTM으로 다음 문자를 예측합니다. 문자 ID 생성부터 입력과 정답 구성, 모델 학습, Temperature를 이용한 반복 생성까지 하나의 Colab 실습으로 연결합니다.

## 학습 목표

- 문자와 문자 ID의 관계를 설명할 수 있습니다.
- 연속된 문자 입력과 다음 문자 정답을 만들 수 있습니다.
- Embedding, LSTM, Dense층의 역할을 구분할 수 있습니다.
- Keras로 Character-level LSTM을 학습할 수 있습니다.
- 예측한 문자를 다시 입력에 추가하여 텍스트를 생성할 수 있습니다.
- Temperature에 따른 생성 결과의 차이를 설명할 수 있습니다.

> 이 실습은 자연스러운 Shakespeare 문장을 완성하는 것보다 문자 단위 생성 과정을 이해하는 데 목적이 있습니다. Colab 단일 노트북에서 실행하며 외부 모듈과 사용자 정의 클래스는 사용하지 않습니다.

[Google Colab에서 Lecture 14 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec14_shakespeare.ipynb)

---

## 1. 문자 단위 데이터 준비

- Shakespeare 텍스트에서 사용할 문자 수를 정합니다.
- 각 문자에 정수 ID를 부여합니다.

![문자 단위 데이터 준비](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec14_01_char_data.png)

먼저 Shakespeare 텍스트를 내려받고, 수업 시간에 사용할 앞부분만 선택합니다.

```python
from tensorflow import keras

MAX_CHARS = 5000

path = keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(path, encoding="utf-8").read()[:MAX_CHARS]
```

텍스트에 등장한 고유 문자를 정렬하고 문자와 정수 ID의 대응표를 만듭니다.

```python
chars = sorted(set(text))

char_to_id = {
    char: char_id
    for char_id, char in enumerate(chars)
}

id_to_char = {
    char_id: char
    for char, char_id in char_to_id.items()
}

vocab_size = len(chars)
```

두 사전의 역할은 서로 반대입니다.

| 사전 | 변환 방향 | 사용 시점 |
|---|---|---|
| `char_to_id` | 문자 → ID | 모델 입력 만들기 |
| `id_to_char` | ID → 문자 | 예측 결과를 문자로 복원 |

전체 텍스트도 문자 ID 배열로 변환합니다.

```python
import numpy as np

encoded = np.array([
    char_to_id[char]
    for char in text
])
```

---

## 2. 입력과 다음 문자 정답 만들기

- 연속된 `SEQ_LEN`개 문자를 입력으로 사용합니다.
- 입력 바로 다음 문자가 정답 `y`가 됩니다.

![입력과 다음 문자 정답 만들기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec14_02_input_target.png)

문자 20개를 읽고 바로 다음 문자 하나를 예측하도록 데이터를 구성합니다.

```text
입력 X: 0번째 문자부터 19번째 문자까지
정답 y: 바로 다음인 20번째 문자
```

```python
SEQ_LEN = 20

X = np.array([
    encoded[i:i + SEQ_LEN]
    for i in range(len(encoded) - SEQ_LEN)
])

y = np.array([
    encoded[i + SEQ_LEN]
    for i in range(len(encoded) - SEQ_LEN)
])
```

첫 번째 입력과 정답을 문자로 다시 확인하면 모델이 무엇을 학습하는지 분명해집니다.

```python
input_text = "".join(
    id_to_char[char_id]
    for char_id in X[0]
)

target_char = id_to_char[y[0]]

print("입력:", repr(input_text))
print("정답:", repr(target_char))
```

`X`의 shape은 `(샘플 수, SEQ_LEN)`이고 `y`의 shape은 `(샘플 수,)`입니다.

---

## 3. Character-level LSTM 모델

- Embedding은 문자 ID를 벡터로 바꿉니다.
- LSTM은 문맥을 읽고 다음 문자 확률을 계산합니다.

![Character-level LSTM 모델](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec14_03_lstm_model.png)

이 실습은 초보자가 모델 흐름을 쉽게 확인할 수 있도록 `keras.Sequential`을 사용합니다.

```python
from tensorflow.keras import layers

EMBED_DIM = 32
LSTM_UNITS = 64

model = keras.Sequential([
    layers.Input(shape=(SEQ_LEN,)),
    layers.Embedding(vocab_size, EMBED_DIM),
    layers.LSTM(LSTM_UNITS),
    layers.Dense(vocab_size, activation="softmax")
])
```

```text
문자 ID 20개
   ↓
Embedding 벡터 20개
   ↓
LSTM의 마지막 출력
   ↓
각 문자 후보의 확률
```

이번 실습은 입력 시퀀스 전체에서 다음 문자 하나를 예측하므로 `LSTM`의 마지막 출력만 사용합니다. 따라서 `return_sequences=True`를 지정하지 않습니다.

---

## 4. 모델 학습하기

- `compile`에서 손실 함수와 optimizer를 정합니다.
- `fit`으로 `X`에서 다음 문자 `y`를 학습합니다.

![Character-level LSTM 모델 학습](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec14_04_model_training.png)

정답은 문자 ID 하나이므로 `sparse_categorical_crossentropy`를 사용합니다.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

```python
EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)
```

`MAX_CHARS=5000`, `LSTM_UNITS=64`, `EPOCHS=10`은 수업 중 빠른 실행을 위한 설정입니다. 생성 품질을 높이려면 데이터, 모델 크기, 학습 횟수를 늘릴 수 있지만 실행 시간도 증가합니다.

---

## 5. 다음 문자를 반복해서 생성하기

- 예측한 문자를 입력 뒤에 붙이고 다시 예측합니다.
- Temperature로 안정성과 다양성을 조절합니다.

![다음 문자를 반복해서 생성하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec14_05_text_generation.png)

텍스트 생성은 다음 과정을 반복합니다.

```text
최근 20문자 입력
      ↓
다음 문자 확률 계산
      ↓
Temperature 적용
      ↓
문자 하나 선택
      ↓
선택한 문자를 입력 뒤에 추가
```

```python
def generate_text(seed, length=200, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature는 0보다 커야 합니다.")

    result = list(seed)
    sequence = [
        char_to_id[char]
        for char in seed[-SEQ_LEN:]
    ]

    for _ in range(length):
        model_input = np.array([
            sequence[-SEQ_LEN:]
        ])

        probabilities = model.predict(
            model_input,
            verbose=0
        )[0]

        log_probs = np.log(probabilities + 1e-10)
        scaled_probs = log_probs / temperature
        scaled_probs = np.exp(scaled_probs)
        scaled_probs = scaled_probs / scaled_probs.sum()

        next_id = np.random.choice(
            len(scaled_probs),
            p=scaled_probs
        )

        result.append(id_to_char[next_id])
        sequence.append(next_id)

    return "".join(result[len(seed):])
```

여러 Temperature를 적용해 생성 결과를 비교합니다.

```python
seed = text[:SEQ_LEN]

for temperature in [0.2, 0.5, 1.0, 1.5]:
    generated = generate_text(
        seed,
        length=150,
        temperature=temperature
    )

    print(f"\nTemperature = {temperature}")
    print(generated)
```

| Temperature | 일반적인 경향 |
|---:|---|
| 낮음 | 확률이 높은 문자를 자주 선택하여 안정적이지만 반복적일 수 있음 |
| 높음 | 다양한 문자를 선택하지만 문맥이 불안정해질 수 있음 |

---

## 핵심 정리

- 문자 단위 모델은 문자를 ID로 변환한 뒤 다음 문자를 예측합니다.
- `X`는 연속된 문자 시퀀스이고 `y`는 바로 다음 문자입니다.
- LSTM은 문자 순서를 읽어 문맥을 하나의 출력으로 요약합니다.
- 생성에서는 예측 문자를 입력에 추가하고 같은 과정을 반복합니다.
- Temperature는 생성의 안정성과 다양성을 조절합니다.

## 실습 파일

- [Google Colab에서 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec14_shakespeare.ipynb)
- 노트북 파일: `lec14_shakespeare.ipynb`

