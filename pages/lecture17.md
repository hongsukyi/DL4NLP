# Lecture 17. Decoder-only Transformer 문자 생성

이번 강의에서는 Lecture 14와 같은 Shakespeare 문자 데이터를 사용하고, LSTM 대신 작은 Decoder-only Transformer를 학습합니다. Causal Self-Attention, Positional Encoding, Feed-Forward Network를 결합하여 모든 위치의 다음 문자를 학습하고 텍스트를 반복 생성합니다.

## 학습 목표

- 문자 시퀀스와 한 칸 이동한 정답을 만들 수 있습니다.
- 문자 임베딩에 Sin/Cos 위치 정보를 추가할 수 있습니다.
- Decoder Transformer block의 구성 요소를 설명할 수 있습니다.
- Causal mask가 필요한 이유를 설명할 수 있습니다.
- 모든 위치에서 다음 문자를 학습할 수 있습니다.
- 마지막 위치의 logits를 이용해 문자를 반복 생성할 수 있습니다.
- LSTM과 Transformer의 문자 생성 방식을 비교할 수 있습니다.

> 이 실습은 작은 데이터와 모델을 사용합니다. 자연스러운 Shakespeare 문장을 완성하는 것보다 Decoder-only Transformer의 학습과 생성 과정을 이해하는 데 목적이 있습니다.

[Google Colab에서 Lecture 17 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec17_decoder_shakespeare.ipynb)

---

## 1. 문자 입력과 다음 문자 정답 만들기

- Shakespeare 텍스트를 문자 ID로 변환합니다.
- 입력 `X`보다 정답 `y`를 한 문자 앞으로 이동합니다.

![문자 입력과 다음 문자 정답 만들기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec17_01_shifted_characters.png)

텍스트에 등장한 고유 문자로 문자 사전을 만듭니다.

```python
text = open(
    path,
    encoding="utf-8"
).read()[:MAX_CHARS]

chars = sorted(set(text))

char_to_id = {
    char: idx
    for idx, char in enumerate(chars)
}

id_to_char = {
    idx: char
    for char, idx in char_to_id.items()
}

encoded = np.array([
    char_to_id[char]
    for char in text
])
```

Lecture 14에서는 문자 20개로 다음 문자 하나를 예측했습니다. Lecture 17에서는 모든 입력 위치에 대응하는 다음 문자 정답을 만듭니다.

```text
입력 X: T  h  e     K  i  n  g
정답 y: h  e     K  i  n  g  ...
```

```python
X = np.array([
    encoded[i:i + SEQ_LEN]
    for i in range(len(encoded) - SEQ_LEN)
])

y = np.array([
    encoded[i + 1:i + SEQ_LEN + 1]
    for i in range(len(encoded) - SEQ_LEN)
])
```

`X`와 `y`는 모두 `(샘플 수, SEQ_LEN)`의 shape을 갖습니다.

---

## 2. 문자 임베딩에 위치 정보 더하기

- 문자 ID를 `D_MODEL` 차원의 벡터로 바꿉니다.
- Sin/Cos PE를 더해 각 문자의 순서를 표현합니다.

![문자 임베딩에 위치 정보 더하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec17_02_embedding_pe.png)

문자 ID는 Embedding층을 통해 학습 가능한 벡터로 변환됩니다.

```python
token_input = keras.Input(
    shape=(SEQ_LEN,),
    dtype="int32",
    name="characters"
)

x = layers.Embedding(
    input_dim=vocab_size,
    output_dim=D_MODEL,
    name="character_embedding"
)(token_input)
```

Lecture 16에서 만든 Sin/Cos PE를 문자 임베딩에 더합니다.

```python
pe_matrix = build_sinusoidal_pe(
    SEQ_LEN,
    D_MODEL
)

x = keras.ops.add(
    x,
    pe_matrix
)
```

```text
문자 임베딩 → 문자의 종류를 표현
Sin/Cos PE  → 문자의 위치를 표현
두 벡터의 합 → Transformer 입력
```

---

## 3. Decoder Transformer Block 구성하기

- Causal mask로 미래 문자를 보지 못하게 합니다.
- Attention 뒤에 FFN과 잔차 연결·정규화를 적용합니다.

![Decoder Transformer Block 구성하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec17_03_decoder_block.png)

Decoder block은 다음 순서로 데이터를 처리합니다.

```text
Transformer 입력
→ Causal Self-Attention
→ 잔차 연결
→ Layer Normalization
→ Feed-Forward Network
→ 잔차 연결
→ Layer Normalization
```

Self-Attention에는 `use_causal_mask=True`를 적용합니다.

```python
attention_output = layers.MultiHeadAttention(
    num_heads=N_HEADS,
    key_dim=D_MODEL // N_HEADS,
    dropout=DROPOUT
)(
    x,
    x,
    use_causal_mask=True
)
```

현재 위치는 자신과 이전 문자만 참고할 수 있고 미래 문자는 볼 수 없습니다. Causal mask가 없으면 모델이 학습 중 미래 정답을 미리 볼 수 있습니다.

Attention 출력에는 잔차 연결과 정규화를 적용합니다.

```python
attention_output = layers.Dropout(
    DROPOUT
)(attention_output)

x = layers.Add()([
    x,
    attention_output
])

x = layers.LayerNormalization(
    epsilon=1e-6
)(x)
```

이후 각 위치에 같은 Feed-Forward Network를 적용합니다.

```python
ffn_output = layers.Dense(
    FF_DIM,
    activation="relu"
)(x)

ffn_output = layers.Dense(
    D_MODEL
)(ffn_output)

x = layers.Add()([
    x,
    ffn_output
])

x = layers.LayerNormalization(
    epsilon=1e-6
)(x)
```

문자 시퀀스의 길이가 모두 `SEQ_LEN`으로 같으므로 이 실습에는 Padding mask가 필요하지 않습니다.

---

## 4. 모든 위치에서 다음 문자 학습하기

- 모델은 각 위치마다 다음 문자의 logits를 출력합니다.
- 한 시퀀스에서 여러 개의 다음 문자 정답을 학습합니다.

![모든 위치에서 다음 문자 학습하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec17_04_all_positions.png)

마지막 위치 하나만 선택하지 않고 모든 위치의 출력을 유지합니다.

```python
output = layers.Dense(
    vocab_size,
    name="next_character"
)(x)

model = keras.Model(
    token_input,
    output
)
```

모델과 데이터 shape의 관계는 다음과 같습니다.

```text
입력 X : (batch, sequence)
출력   : (batch, sequence, vocabulary)
정답 y : (batch, sequence)
```

각 위치의 정답은 정수 문자 ID이므로 `SparseCategoricalCrossentropy`를 사용합니다.

```python
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["sparse_categorical_accuracy"]
)

history = model.fit(
    X,
    y,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=2
)
```

한 시퀀스 안의 여러 위치에서 다음 문자 학습이 동시에 이루어지는 것이 표준적인 Decoder 언어모델 방식입니다.

---

## 5. 마지막 위치에서 다음 문자 생성하기

- 현재 문맥의 마지막 위치 logits에서 문자를 선택합니다.
- 선택한 문자를 문맥에 붙이고 예측을 반복합니다.

![마지막 위치에서 다음 문자 생성하기](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec17_05_generation_loop.png)

학습할 때는 모든 위치의 출력을 사용하지만 생성할 때는 현재 문맥의 마지막 위치만 사용합니다.

```python
context = np.array([
    sequence[-SEQ_LEN:]
])

logits = model.predict(
    context,
    verbose=0
)[0, -1]
```

Temperature를 적용한 뒤 다음 문자 하나를 선택합니다.

```python
scaled_logits = logits / temperature

probabilities = tf.nn.softmax(
    scaled_logits
).numpy()

next_id = np.random.choice(
    vocab_size,
    p=probabilities
)
```

선택한 문자 ID를 문맥 뒤에 추가하고 같은 과정을 반복합니다.

```python
sequence.append(next_id)
generated_chars.append(
    id_to_char[next_id]
)
```

```text
학습 → 모든 위치에서 다음 문자 예측
생성 → 마지막 위치에서 다음 문자 선택
```

Temperature가 낮으면 확률이 높은 문자를 자주 선택하고, 높으면 더 다양한 문자를 선택합니다.

---

## Lecture 14 LSTM과 비교

| 항목 | Lecture 14 LSTM | Lecture 17 Transformer |
|---|---|---|
| 문맥 처리 | 은닉 상태를 순서대로 갱신 | 이전 위치들을 Self-Attention으로 참고 |
| 위치 정보 | 순차 처리 자체에 포함 | Positional Encoding 추가 |
| 미래 차단 | 순차 구조에서 자연스럽게 차단 | Causal mask 필요 |
| 학습 출력 | 시퀀스 다음 문자 하나 | 모든 위치의 다음 문자 |
| 생성 | 예측 문자를 다시 입력 | 예측 문자를 다시 입력 |

## 핵심 정리

- Decoder-only Transformer는 Causal Self-Attention으로 이전 문자만 참고합니다.
- Positional Encoding은 문자 순서 정보를 추가합니다.
- 학습에서는 모든 위치의 다음 문자를 동시에 예측합니다.
- 생성에서는 현재 문맥의 마지막 위치 logits만 사용합니다.
- 선택한 문자를 문맥에 추가하면서 자기회귀 생성을 반복합니다.

## 실습 파일

- [Google Colab에서 실행](https://colab.research.google.com/github/hongsukyi/DL4NLP/blob/main/lec17_decoder_shakespeare.ipynb)
- 노트북 파일: `lec17_decoder_shakespeare.ipynb`

