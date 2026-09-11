# Lecture 15. Subword Self-Attention 실습

이번 강의에서는 한국어 문장을 서브워드로 나누고, 작은 Self-Attention 모델로 각 위치의 다음 서브워드를 예측합니다. 토큰 임베딩과 위치 임베딩, Padding mask와 Causal mask, 패딩 loss 제외, Attention 가중치 확인 과정을 하나의 Colab 실습으로 구성합니다.

## 학습 목표

- KLUE 토크나이저로 한국어 문장을 서브워드로 나눌 수 있습니다.
- 실습 데이터에 등장한 서브워드로 작은 지역 어휘를 만들 수 있습니다.
- 입력과 정답을 한 칸 이동하여 다음 토큰 예측 데이터를 만들 수 있습니다.
- 토큰 임베딩과 위치 임베딩의 역할을 구분할 수 있습니다.
- Padding mask와 Causal mask의 차이를 설명할 수 있습니다.
- 다음 서브워드 예측과 Attention 가중치를 확인할 수 있습니다.

> 이 실습은 KLUE-BERT를 학습하는 것이 아닙니다. KLUE 토크나이저는 서브워드 분리에만 사용하며, 실습 데이터로 작은 Self-Attention 모델을 처음부터 학습합니다.

[Google Colab에서 Lecture 15 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec15_self_attention_v2.ipynb)

---

## 1. 서브워드와 작은 어휘 만들기

- KLUE 토크나이저로 문장을 서브워드로 나눕니다.
- 실습 문장에 등장한 서브워드만 사전에 넣습니다.

![서브워드와 작은 어휘 만들기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec15_01_subword_vocab.png)

KLUE 토크나이저를 불러오고 각 문장을 서브워드로 나눕니다. 문장의 끝을 표시하기 위해 `[EOS]`를 추가합니다.

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

KLUE의 전체 어휘를 출력층으로 사용하면 작은 실습에 비해 모델이 지나치게 커집니다. 따라서 실습 문장에 실제로 등장한 서브워드만 모아 지역 어휘를 만듭니다.

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
```

| 특수 토큰 | 역할 |
|---|---|
| `[PAD]` | 서로 다른 문장 길이를 맞춤 |
| `[UNK]` | 지역 어휘에 없는 토큰을 표현 |
| `[EOS]` | 문장의 끝을 표시 |

---

## 2. 입력과 정답을 한 칸 이동하기

- 입력 `X`의 다음 서브워드가 정답 `y`가 됩니다.
- `sample_weight`로 `[PAD]` 위치를 학습에서 제외합니다.

![입력과 정답을 한 칸 이동하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec15_02_shift_padding.png)

각 문장을 ID로 변환하고 최대 길이에 맞춰 `[PAD]`를 추가합니다. 그다음 입력과 정답을 한 칸 이동합니다.

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

Attention에서 `[PAD]`를 보지 않게 하는 것과 `[PAD]` 정답을 loss에서 제외하는 것은 서로 다른 작업입니다.

```python
sample_weight = (
    y != pad_id
).astype("float32")
```

```text
sample_weight = 1 → 실제 서브워드이므로 loss 계산
sample_weight = 0 → [PAD]이므로 loss에서 제외
```

---

## 3. 토큰 임베딩과 위치 임베딩

- 토큰 임베딩은 서브워드의 의미 표현을 만듭니다.
- 위치 임베딩은 문장 속 순서를 알려줍니다.

![토큰 임베딩과 위치 임베딩](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec15_03_token_position.png)

Self-Attention은 토큰의 순서를 자동으로 알지 못하므로 위치 정보를 별도로 추가해야 합니다.

```python
position_ids = np.tile(
    np.arange(SEQ_LEN, dtype=np.int32),
    (len(X), 1)
)
```

토큰 ID와 위치 ID를 각각 임베딩하고 두 벡터를 더합니다.

```python
token_input = keras.Input(
    shape=(SEQ_LEN,),
    dtype="int32",
    name="tokens"
)

position_input = keras.Input(
    shape=(SEQ_LEN,),
    dtype="int32",
    name="positions"
)

token_embedding = layers.Embedding(
    vocab_size,
    D_MODEL,
    name="token_embedding"
)(token_input)

position_embedding = layers.Embedding(
    SEQ_LEN,
    D_MODEL,
    name="position_embedding"
)(position_input)

x = layers.Add()([
    token_embedding,
    position_embedding
])
```

같은 서브워드라도 위치가 다르면 서로 다른 입력 표현을 갖게 됩니다.

---

## 4. Self-Attention에 두 마스크 적용

- Padding mask는 `[PAD]` 토큰을 보지 못하게 합니다.
- Causal mask는 미래 토큰을 보지 못하게 합니다.

![Self-Attention에 Padding mask와 Causal mask 적용](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec15_04_attention_masks.png)

Padding mask는 실제 토큰과 `[PAD]`를 구분합니다.

```python
padding_mask = keras.ops.not_equal(
    token_input,
    pad_id
)

padding_mask = keras.ops.expand_dims(
    padding_mask,
    axis=1
)
```

Causal mask는 현재 위치가 미래 정답을 미리 보지 못하게 합니다. Keras에서는 `use_causal_mask=True`로 적용할 수 있습니다.

```python
attention_layer = layers.MultiHeadAttention(
    num_heads=N_HEADS,
    key_dim=D_MODEL // N_HEADS,
    name="self_attention"
)

attention_output, attention_scores = attention_layer(
    query=x,
    value=x,
    key=x,
    attention_mask=padding_mask,
    use_causal_mask=True,
    return_attention_scores=True
)
```

```text
Padding mask → 문장 뒤의 [PAD]를 차단
Causal mask  → 현재 위치보다 뒤의 미래 토큰을 차단
```

두 마스크는 목적이 다르므로 다음 서브워드 예측에서는 함께 필요합니다.

---

## 5. 다음 서브워드 예측과 Attention 확인

- 각 위치에서 실제 정답과 예측 토큰을 비교합니다.
- Attention 가중치는 참고한 토큰의 비중을 보여줍니다.

![다음 서브워드 예측과 Attention 확인](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec15_05_prediction_attention.png)

Attention 출력에 잔차 연결과 정규화를 적용한 뒤 각 위치의 다음 토큰 logits를 만듭니다.

```python
x = layers.Add()([
    x,
    attention_output
])

x = layers.LayerNormalization()(x)

logits = layers.Dense(
    vocab_size,
    name="next_token"
)(x)
```

패딩 정답은 `sample_weight`를 전달하여 loss와 accuracy에서 제외합니다.

```python
history = train_model.fit(
    [X, position_ids],
    y,
    sample_weight=sample_weight,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)
```

학습 후 각 위치의 정답과 예측을 비교합니다.

```python
logits = train_model.predict(
    [X[:1], position_ids[:1]],
    verbose=0
)

predicted_ids = logits.argmax(axis=-1)[0]
```

Attention 가중치는 한 위치가 앞의 어느 토큰 정보를 상대적으로 많이 결합했는지 보여줍니다. 하지만 가중치가 높다는 사실만으로 모델이 언어적 의미나 대명사 관계를 이해했다고 단정할 수는 없습니다.

> Attention 가중치는 모델 내부 정보 결합을 관찰하는 참고 자료이며, 그 자체가 완전한 설명이나 언어 이해의 증거는 아닙니다.

---

## 핵심 정리

- KLUE 토크나이저는 한국어 문장을 서브워드로 분리하는 데 사용합니다.
- 실습에 등장한 토큰만 사용하면 작고 빠른 지역 어휘를 만들 수 있습니다.
- 입력과 정답을 한 칸 이동하여 모든 위치의 다음 서브워드를 학습합니다.
- 토큰 임베딩에는 의미 정보, 위치 임베딩에는 순서 정보가 담깁니다.
- Padding mask, Causal mask, `sample_weight`는 서로 다른 문제를 해결합니다.
- Attention 가중치는 참고한 토큰의 비중으로 제한적으로 해석해야 합니다.

## 실습 파일

- [Google Colab에서 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec15_self_attention_v2.ipynb)
- 노트북 파일: `lec15_self_attention_v2.ipynb`

