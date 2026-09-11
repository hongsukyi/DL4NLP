# Lecture 16. Positional Encoding 비교 실습

이번 강의에서는 Lecture 15의 Self-Attention 모델을 바탕으로 Positional Encoding(PE)이 없는 모델과 있는 모델을 비교합니다. 두 모델의 데이터, 구조, 초기 가중치와 학습 조건을 동일하게 맞추고 PE 사용 여부만 변경하여 학습 loss의 변화를 관찰합니다.

## 학습 목표

- 실습 문장으로 작은 서브워드 지역 어휘를 만들 수 있습니다.
- 다음 토큰 예측을 위한 입력과 정답을 구성할 수 있습니다.
- `[PAD]` 정답을 loss에서 제외할 수 있습니다.
- Sin/Cos Positional Encoding을 만들 수 있습니다.
- 동일한 초기 가중치로 두 모델을 공정하게 비교할 수 있습니다.
- 작은 toy 실험의 결과를 제한적으로 해석할 수 있습니다.

> 이 실습의 목적은 PE가 항상 더 낮은 loss를 보장한다고 결론 내리는 것이 아닙니다. 위치 정보를 추가했을 때 학습 양상이 어떻게 달라지는지 관찰하는 데 목적이 있습니다.

[Google Colab에서 Lecture 16 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec16_positional_encoding.ipynb)

---

## 1. 고유 문장과 지역 어휘 준비

- 30개 고유 문장을 KLUE 토크나이저로 분리합니다.
- 실습에 등장한 서브워드만 작은 어휘로 만듭니다.

![고유 문장과 지역 어휘 준비](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec16_01_local_vocab.png)

Lecture 15와 같은 30개 고유 문장을 사용합니다. 같은 문장을 여러 번 복제해도 새로운 정보가 추가되지 않으므로 문장 반복은 사용하지 않습니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "klue/bert-base"
)

token_lists = []

for sentence in sentences:
    tokens = tokenizer.tokenize(sentence)
    tokens.append("[EOS]")
    token_lists.append(tokens)
```

KLUE 전체 어휘 대신 실습 문장에 등장한 서브워드만 사용합니다.

```python
all_tokens = sorted({
    token
    for tokens in token_lists
    for token in tokens
})

token_to_id = {
    "[PAD]": 0,
    "[UNK]": 1
}

for token in all_tokens:
    if token not in token_to_id:
        token_to_id[token] = len(token_to_id)

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}

pad_id = token_to_id["[PAD]"]
vocab_size = len(token_to_id)
```

이 모델은 KLUE-BERT가 아닙니다. KLUE 토크나이저로 문장을 나눈 뒤, 작은 Self-Attention 모델을 처음부터 학습합니다.

---

## 2. 입력·정답 시프트와 패딩 제외

- 입력 `X`보다 정답 `y`를 한 토큰 앞으로 이동합니다.
- `[PAD]` 위치는 `sample_weight=0`으로 loss에서 제외합니다.

![입력 정답 시프트와 패딩 제외](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec16_02_shift_padding.png)

다음 토큰 예측에서는 입력과 정답이 한 칸 이동합니다.

```text
입력 X: 고양이  ##가  매트  위  ##에
정답 y: ##가    매트  위    ##에  앉
```

```python
max_tokens = max(
    len(tokens)
    for tokens in token_lists
)

SEQ_LEN = max_tokens - 1

X = []
y = []

for tokens in token_lists:
    ids = [
        token_to_id[token]
        for token in tokens
    ]

    ids = ids + [pad_id] * (
        max_tokens - len(ids)
    )

    X.append(ids[:-1])
    y.append(ids[1:])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)
```

`[PAD]`는 실제 다음 토큰 정답이 아니므로 loss 계산에서 제외합니다.

```python
sample_weight = (
    y != pad_id
).astype("float32")
```

Attention에서 `[PAD]`를 보지 않게 하는 Padding mask와 `[PAD]` 정답을 loss에서 제외하는 `sample_weight`는 서로 다른 역할을 합니다.

---

## 3. Sin/Cos 위치 인코딩 만들기

- 토큰의 위치마다 서로 다른 sin과 cos 값을 계산합니다.
- 미리 만든 PE 행렬을 토큰 임베딩에 더합니다.

![Sin Cos 위치 인코딩 만들기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec16_03_sinusoidal_pe.png)

Self-Attention만으로는 토큰의 절대적인 순서를 직접 알 수 없습니다. Sin/Cos PE는 각 위치에 서로 다른 반복 패턴을 부여합니다.

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

```python
def build_sinusoidal_pe(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    angles = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model
    )

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles.astype("float32")

pe_matrix = build_sinusoidal_pe(
    SEQ_LEN,
    D_MODEL
)
```

PE를 사용하는 모델에서는 토큰 임베딩에 상수 행렬을 더합니다.

```python
x = keras.ops.add(
    x,
    pe_matrix
)
```

---

## 4. 동일 초기 가중치로 두 모델 만들기

- 두 모델은 PE 사용 여부만 다르게 구성합니다.
- `set_weights()`로 같은 초기 가중치에서 시작합니다.

![동일 초기 가중치로 두 모델 만들기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec16_04_same_weights.png)

두 모델은 다음 구성 요소를 동일하게 사용합니다.

- 토큰 Embedding
- Multi-Head Self-Attention
- Padding mask와 Causal mask
- 잔차 연결과 Layer Normalization
- 다음 토큰 출력층

```python
model_no_pe = build_model(
    use_pe=False,
    model_name="without_pe"
)

model_with_pe = build_model(
    use_pe=True,
    model_name="with_pe"
)
```

모델을 따로 생성하면 처음에는 서로 다른 무작위 가중치를 가질 수 있습니다. 비교 전에 PE 모델의 가중치를 PE가 없는 모델과 같게 맞춥니다.

```python
model_with_pe.set_weights(
    model_no_pe.get_weights()
)
```

초기 가중치가 다르면 PE의 효과와 무작위 초기화의 효과를 구분하기 어렵습니다.

---

## 5. 같은 조건으로 학습 비교하기

- 같은 `X`, `y`, `sample_weight`와 배치 순서를 사용합니다.
- 학습 후 loss 변화만 비교하고 과도하게 해석하지 않습니다.

![같은 조건으로 학습 비교하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec16_05_fair_training.png)

두 모델에는 같은 데이터와 학습 조건을 전달합니다.

```python
history_no_pe = model_no_pe.fit(
    X,
    y,
    sample_weight=sample_weight,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=2
)

history_with_pe = model_with_pe.fit(
    X,
    y,
    sample_weight=sample_weight,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=2
)
```

공정한 비교를 위해 다음 조건을 동일하게 유지합니다.

| 조건 | PE 없음 | PE 있음 |
|---|---|---|
| 입력과 정답 | 동일 | 동일 |
| 초기 가중치 | 동일 | 동일 |
| 모델 구조 | 동일 | 동일 |
| Epoch와 Batch | 동일 | 동일 |
| 데이터 순서 | 동일 | 동일 |
| Positional Encoding | 사용하지 않음 | 사용 |

30개 문장을 사용한 toy 실험이므로 최종 loss 차이를 일반적인 성능 우위로 확대 해석하지 않습니다. 여러 seed와 더 큰 데이터에서 반복해야 더 신뢰할 수 있는 결론을 얻을 수 있습니다.

---

## 핵심 정리

- 위치 인코딩은 Self-Attention에 토큰 순서 정보를 제공합니다.
- PE 유무를 비교할 때는 초기 가중치와 학습 조건을 동일하게 맞춰야 합니다.
- Padding mask와 `sample_weight`는 서로 다른 문제를 해결합니다.
- 작은 toy 데이터의 loss 차이는 학습 경향을 관찰하는 자료로만 사용합니다.

## 실습 파일

- [Google Colab에서 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec16_positional_encoding.ipynb)
- 노트북 파일: `lec16_positional_encoding.ipynb`

