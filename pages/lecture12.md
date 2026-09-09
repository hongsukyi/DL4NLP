# Lecture 12. Character-level LSTM 텍스트 생성

이번 강의에서는 문장을 문자 단위로 나누고 LSTM으로 다음 문자를 예측합니다. 작은 `hello` 예제를 통해 입력과 목표의 한 칸 이동, Embedding, LSTM, Dense, Softmax의 역할을 살펴본 뒤 Greedy Decoding과 확률적 Sampling을 비교합니다.

## 학습 목표

- 문자 단위 생성 모델의 장단점을 설명할 수 있습니다.
- 입력 문자열과 다음 문자 목표를 만들 수 있습니다.
- 문자와 문자 ID 사이를 변환할 수 있습니다.
- Keras로 Character-level LSTM 모델을 구성할 수 있습니다.
- `return_sequences=True`가 필요한 이유를 설명할 수 있습니다.
- Greedy, Temperature, Top-k, Top-p 방식을 구분할 수 있습니다.
- 생성 결과의 반복과 문맥 이탈을 해석할 수 있습니다.

> 이 장에서는 Keras 코드를 기본으로 사용합니다. PyTorch 버전은 추후 별도의 링크로 제공합니다.

---

## 1. 문자 단위 생성 모델

문자 단위 모델은 단어나 서브워드 대신 개별 문자를 입력받아 바로 다음 문자를 예측합니다.

![문자를 하나씩 예측하여 텍스트를 만드는 자기회귀 생성](assets/lec12_01_character_generation.png)

```text
입력 h → 다음 문자 e
입력 e → 다음 문자 l
입력 l → 다음 문자 l
입력 l → 다음 문자 o
```

추론에서는 생성한 문자를 다시 입력에 추가합니다.

```text
h → he → hel → hell → hello
```

문자 단위 모델은 자기회귀 생성 원리를 작은 예제로 확인하기에 적합합니다.

---

## 2. 문자 단위 모델의 장단점

![작은 문자 집합과 길어진 시퀀스로 보는 문자 모델의 장단점](assets/lec12_02_character_model_pros_cons.png)

| 장점 | 단점 |
|---|---|
| 문자 집합이 비교적 작음 | 단어보다 시퀀스가 길어짐 |
| 처음 보는 단어도 문자 조합으로 표현 | 순차 계산 횟수가 증가 |
| 신조어와 오타를 표현할 수 있음 | 단어 의미를 직접 표현하기 어려움 |
| 토큰 밖 단어 문제가 거의 없음 | 긴 문맥을 유지하기 어려움 |

예를 들어 `chatbot`은 단어 토큰 하나가 될 수 있지만 문자 단위에서는 일곱 시점으로 나뉩니다.

```text
단어 단위: [chatbot]
문자 단위: [c, h, a, t, b, o, t]
```

---

## 3. `hello` 학습 데이터 만들기

다음 문자 예측 데이터는 원문을 한 칸 이동하여 만듭니다.

![hello 문자열에서 입력과 다음 문자 목표를 만드는 과정](assets/lec12_03_hello_training_pairs.png)

```text
원문: h e l l o
입력: h e l l
목표: e l l o
```

각 시점의 학습 쌍은 다음과 같습니다.

| 시점 | 입력 문자 | 목표 문자 |
|---:|---|---|
| 1 | h | e |
| 2 | e | l |
| 3 | l | l |
| 4 | l | o |

```python
text = "hello"

input_chars = list(text[:-1])
target_chars = list(text[1:])

print(input_chars)
print(target_chars)
```

```text
['h', 'e', 'l', 'l']
['e', 'l', 'l', 'o']
```

---

## 4. 문자 ID와 Embedding

신경망은 문자를 직접 계산하지 못하므로 각 문자를 고유한 정수 ID로 바꿉니다.

![문자를 ID로 변환하고 Embedding 벡터를 찾는 과정](assets/lec12_04_character_ids_embedding.png)

```python
chars = sorted(set(text))

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

input_ids = [char_to_id[ch] for ch in input_chars]
target_ids = [char_to_id[ch] for ch in target_chars]

print(char_to_id)
print(input_ids)
print(target_ids)
```

Embedding층은 각 문자 ID를 학습 가능한 밀집 벡터로 변환합니다.

```text
문자 → 문자 ID → Embedding 벡터
```

같은 문자는 어느 위치에서 등장하더라도 같은 ID를 사용합니다. 하지만 LSTM의 은닉 상태가 앞의 문맥을 반영하므로 같은 문자도 문맥에 따라 다른 출력 상태를 만들 수 있습니다.

---

## 5. Character-level LSTM 구조

![Embedding LSTM Dense로 구성된 문자 단위 생성 모델](assets/lec12_05_lstm_architecture.png)

모델의 데이터 흐름은 다음과 같습니다.

```text
문자 ID
   ↓
Embedding
   ↓
LSTM(return_sequences=True)
   ↓
Dense(문자 수)
   ↓
각 시점의 다음 문자 logits
```

```python
vocab_size = len(chars)
embedding_dim = 8
hidden_size = 32

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(vocab_size, embedding_dim)(inputs)
x = layers.LSTM(hidden_size, return_sequences=True)(x)
logits = layers.Dense(vocab_size)(x)

model = keras.Model(inputs, logits)
```

`return_sequences=True`는 LSTM이 마지막 시점 하나가 아니라 모든 시점의 출력을 반환하도록 합니다.

| 설정 | LSTM 출력 | 용도 |
|---|---|---|
| `return_sequences=False` | 마지막 출력 하나 | 문장 분류 |
| `return_sequences=True` | 모든 시점의 출력 | 다음 문자 예측 |

```text
LSTM 출력  : [batch, time, hidden_size]
Dense 출력 : [batch, time, vocab_size]
```

Keras의 Dense층은 3차원 입력의 마지막 축에 같은 가중치를 적용하므로 각 시점에서 다음 문자 logits를 만듭니다.

---

## 6. Logits와 Softmax

Dense층은 각 문자 후보에 대한 logit을 출력합니다. Logit은 아직 확률이 아닌 실수 점수입니다.

![문자별 logits를 Softmax로 확률 분포로 바꾸는 과정](assets/lec12_06_logits_softmax.png)

$$
P_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

| 문자 후보 | Logit | Softmax 확률 예시 |
|---|---:|---:|
| e | 2.0 | 0.60 |
| l | 1.0 | 0.20 |
| o | 0.5 | 0.10 |
| 기타 | - | 0.10 |

모델 학습에서는 정답 문자 ID와 logits를 이용해 Sparse Categorical Cross-Entropy를 계산합니다.

```python
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

`from_logits=True`는 모델 출력이 Softmax를 통과하지 않은 logits임을 의미합니다.

---

## 7. Greedy Decoding과 확률적 Sampling

다음 문자의 확률 분포가 만들어지면 실제로 사용할 문자 하나를 선택해야 합니다.

![가장 높은 확률을 고르는 Greedy와 확률에 따라 뽑는 Sampling](assets/lec12_07_greedy_sampling.png)

### Greedy Decoding

가장 확률이 높은 문자 하나를 선택합니다.

```python
next_id = int(np.argmax(probabilities))
```

- 같은 입력에서는 항상 같은 결과가 나옵니다.
- 구현이 단순합니다.
- 반복적이고 단조로운 결과가 나올 수 있습니다.

### 확률적 Sampling

각 문자의 확률에 비례하여 다음 문자를 무작위로 선택합니다.

```python
next_id = int(
    np.random.choice(len(probabilities), p=probabilities)
)
```

- 같은 입력에서도 실행할 때마다 결과가 달라질 수 있습니다.
- 다양한 결과를 만들 수 있습니다.
- 확률이 낮은 부자연스러운 문자가 선택될 수도 있습니다.

---

## 8. Temperature, Top-k, Top-p

Temperature와 후보 제한 방법을 사용하면 생성의 다양성을 조절할 수 있습니다.

![Temperature Top-k Top-p를 이용한 다음 문자 선택](assets/lec12_08_sampling_controls.png)

### Temperature

logits를 Temperature $T$로 나눈 뒤 Softmax를 적용합니다.

$$
P_i(T)=\operatorname{softmax}\left(\frac{z_i}{T}\right)
$$

| Temperature | 확률 분포 | 생성 경향 |
|---|---|---|
| 낮음 | 뾰족함 | 안정적이고 반복적 |
| 약 1.0 | 원래 분포와 비슷함 | 균형 있는 생성 |
| 높음 | 평평함 | 다양하지만 불안정할 수 있음 |

### Top-k

확률이 가장 높은 상위 `k`개 문자만 후보로 남깁니다.

```text
Top-k = 3 → 확률 상위 세 문자 중에서 선택
```

### Top-p

확률이 높은 순서로 더했을 때 누적 확률이 `p` 이상이 될 때까지 후보를 남깁니다.

```text
확률: 0.40, 0.25, 0.15, 0.10, ...
Top-p = 0.80 → 앞의 세 후보 사용
```

실제 적용 순서는 다음과 같습니다.

```text
Logits
  ↓ Temperature 적용
조정된 logits
  ↓ Top-k 또는 Top-p로 후보 제한
Softmax와 Sampling
  ↓
다음 문자 선택
```

Top-k와 Top-p는 확장 내용입니다. 초급 실습에서는 Greedy와 Temperature Sampling을 먼저 이해하는 것으로 충분합니다.

---

## 9. 간단한 문자 생성 함수

다음 함수는 현재 문자열의 마지막 출력에서 다음 문자를 선택하고, 이를 입력 끝에 붙이는 과정을 반복합니다.

```python
def generate_text(model, start_text, max_new_chars=20, temperature=1.0):
    generated = list(start_text)

    for _ in range(max_new_chars):
        ids = [char_to_id[ch] for ch in generated]
        x = np.array([ids])

        logits = model.predict(x, verbose=0)[0, -1]
        logits = logits / temperature
        probabilities = tf.nn.softmax(logits).numpy()

        next_id = int(
            np.random.choice(vocab_size, p=probabilities)
        )
        next_char = id_to_char[next_id]
        generated.append(next_char)

    return "".join(generated)
```

```python
print(generate_text(model, "h", max_new_chars=10, temperature=0.8))
```

실제 말뭉치에서는 시작 문자열에 학습 문자 집합에 없는 문자가 들어오는 경우도 처리해야 합니다. 초급 예제에서는 학습에 포함된 문자만 입력한다고 가정합니다.

---

## 10. 생성 결과 해석

좋은 생성 결과는 철자뿐 아니라 문맥의 자연스러움도 유지해야 합니다.

| 관찰 결과 | 가능한 원인 |
|---|---|
| 같은 문자나 단어 반복 | 확률 분포가 일부 후보에 집중 |
| 깨진 단어 | 문자 수준에서 단어 구조를 충분히 학습하지 못함 |
| 문맥 이탈 | 장기 문맥 정보 부족 또는 높은 Temperature |
| 항상 같은 결과 | Greedy Decoding 또는 낮은 Temperature |
| 지나치게 무작위 | 높은 Temperature 또는 작은 학습 데이터 |

결과를 해석할 때는 다음 항목을 함께 확인합니다.

- 훈련 Loss와 검증 Loss
- 생성된 단어의 철자
- 반복되는 문자와 표현
- 문장 전체의 일관성
- Temperature를 바꿨을 때의 변화

문자 단위 LSTM은 생성 원리를 이해하기 위한 교육용 모델입니다. 작은 데이터에서 만든 결과를 현대적인 대규모 언어 모델의 생성 품질과 직접 비교해서는 안 됩니다.

---

## 11. 다음 강의와의 연결

Character-level LSTM은 하나의 모델이 입력을 읽으면서 다음 문자를 계속 생성합니다. 하지만 입력 문장과 출력 문장이 서로 다른 문제, 예를 들어 번역에서는 입력을 처리하는 부분과 출력을 생성하는 부분을 구분할 필요가 있습니다.

```text
입력 문장을 읽는 모델: Encoder
출력 문장을 만드는 모델: Decoder
```

다음 Lecture 13에서는 Encoder–Decoder 구조와 하나의 고정된 Context Vector에 전체 입력을 압축할 때 발생하는 정보 병목 문제를 살펴봅니다.

---

## 핵심 정리

- 문자 단위 모델은 개별 문자로 다음 문자 예측을 학습합니다.
- 입력과 목표는 원문을 한 칸 이동하여 만듭니다.
- Embedding은 문자 ID를 학습 가능한 벡터로 변환합니다.
- 모든 시점에서 다음 문자를 예측하려면 `return_sequences=True`가 필요합니다.
- Dense층은 각 시점의 LSTM 출력에서 문자별 logits를 만듭니다.
- Greedy는 가장 높은 확률의 문자를, Sampling은 확률에 따라 문자를 선택합니다.
- Temperature, Top-k, Top-p는 생성의 안정성과 다양성을 조절합니다.
- 문자 단위 LSTM은 자기회귀 생성 원리를 이해하기 위한 간단한 교육 모델입니다.
