# Lecture 10. Keras로 비교하는 SimpleRNN, LSTM, GRU 실습

이번 실습에서는 Lecture 04에서 저장한 동일한 데이터로 SimpleRNN, LSTM, GRU를 학습합니다. 세 순환 모델을 같은 조건에서 비교하고, Lecture 07에서 얻은 Flatten, Masked GAP, Conv1D 결과와 함께 해석합니다.

## 학습 목표

- 저장된 NPZ 데이터와 고정된 데이터 분할을 불러올 수 있습니다.
- Keras로 SimpleRNN, LSTM, GRU 분류 모델을 만들 수 있습니다.
- 세 순환 모델에 동일한 학습 조건을 적용할 수 있습니다.
- 학습곡선, 테스트 정확도, 혼동행렬을 해석할 수 있습니다.
- 여섯 모델의 결과를 하나의 표와 그래프로 비교할 수 있습니다.
- 검증 데이터와 테스트 데이터의 역할을 구분할 수 있습니다.

> 이 장에서는 Keras 코드를 기본으로 사용합니다. PyTorch 버전은 추후 별도의 링크로 제공합니다.

---

## 1. 실습 전체 흐름

![순환 모델 실습과 종합 비교의 전체 흐름](assets/lec10_01_workflow.png)

```text
저장된 데이터와 기존 결과 불러오기
                ↓
      Embedding과 Masking
                ↓
   SimpleRNN / LSTM / GRU 학습
                ↓
 학습곡선·Accuracy·혼동행렬 확인
                ↓
     기존 세 모델과 종합 비교
```

모든 모델은 같은 훈련·검증·테스트 데이터를 사용합니다. 이번 실행의 결과는 모델 구조를 이해하기 위한 관찰값이며, 난수 시드와 실행 환경이 달라지면 수치도 달라질 수 있습니다.

---

## 2. 데이터와 기존 결과 불러오기

Lecture 04에서 만든 NPZ 파일과 Lecture 07에서 저장한 `res.json`을 사용합니다.

![데이터와 기존 모델 결과를 불러오는 과정](assets/lec10_02_load_data_results.png)

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

```python
data = np.load(
    "aihub_enko_conv_kluebert_3class_n2400_v2.npz",
    allow_pickle=True,
)

ids = data["input_ids"]
mask = data["attention_mask"]
y = data["labels"]

idx_tr = data["train_idx"]
idx_val = data["val_idx"]
idx_te = data["test_idx"]

vocab_size = int(data["vocab_size"])
max_len = int(data["max_len"])
n_cls = int(data["num_classes"])
classes = data["class_names"].tolist()

X_tr, M_tr, y_tr = ids[idx_tr], mask[idx_tr], y[idx_tr]
X_val, M_val, y_val = ids[idx_val], mask[idx_val], y[idx_val]
X_te, M_te, y_te = ids[idx_te], mask[idx_te], y[idx_te]

print("train/val/test:", len(X_tr), len(X_val), len(X_te))
print("classes:", classes)
```

이번 실행에서 확인된 데이터 구성은 다음과 같습니다.

| 항목 | 결과 |
|---|---:|
| 훈련 데이터 | 1,680개 |
| 검증 데이터 | 360개 |
| 테스트 데이터 | 360개 |
| 범주 | 면접, 학교, 협상 |
| 최대 길이 | 32 토큰 |

---

## 3. Embedding과 Masking

토큰 ID를 64차원 임베딩 벡터로 변환하고, `attention_mask`가 0인 PAD 위치를 0으로 만듭니다.

![Embedding과 attention mask를 순환 모델에 적용하는 과정](assets/lec10_03_embedding_masking.png)

```python
dim = 64
units = 64

def embed_mask(input_ids, attention_mask):
    x = layers.Embedding(vocab_size, dim)(input_ids)
    mask_float = layers.Lambda(
        lambda m: tf.cast(m, tf.float32)[..., None]
    )(attention_mask)
    x = layers.Multiply()([x, mask_float])
    return layers.Masking(mask_value=0.0)(x)
```

```text
input_ids       : [batch, 32]
attention_mask  : [batch, 32]
Embedding 출력 : [batch, 32, 64]
```

`Multiply`층은 PAD 위치의 임베딩 벡터를 0으로 만듭니다. 이어지는 `Masking`층은 0벡터인 시점을 마스킹하여 SimpleRNN, LSTM, GRU가 PAD 위치를 건너뛰도록 합니다.

---

## 4. 모델 1: SimpleRNN

SimpleRNN은 현재 입력과 이전 은닉 상태를 이용해 새로운 은닉 상태를 만듭니다.

![Keras SimpleRNN 분류 모델의 구조](assets/lec10_04_simple_rnn.png)

```python
inp = keras.Input((max_len,))
inm = keras.Input((max_len,))

x = embed_mask(inp, inm)
x = layers.SimpleRNN(units)(x)
x = layers.Dense(64, activation="relu")(x)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

rnn_model = keras.Model([inp, inm], out)
```

이번 실행 결과는 다음과 같습니다.

| 항목 | 결과 |
|---|---:|
| 전체 파라미터 | 2,061,507 |
| 테스트 정확도 | 0.636 |
| 가장 낮은 검증 손실 | 0.9327 |
| 해당 Epoch | 2 |

훈련 정확도는 빠르게 높아졌지만 검증 손실은 Epoch 2 이후 다시 증가했습니다. 훈련 데이터에 빠르게 맞춰지면서 일반화 성능이 개선되지 않는 과적합 신호로 볼 수 있습니다.

---

## 5. 모델 2: LSTM

LSTM은 셀 상태와 은닉 상태를 사용하고, 망각·입력·출력 게이트로 정보의 흐름을 조절합니다.

![Keras LSTM 분류 모델의 구조](assets/lec10_05_lstm.png)

```python
inp = keras.Input((max_len,))
inm = keras.Input((max_len,))

x = embed_mask(inp, inm)
x = layers.LSTM(units)(x)
x = layers.Dense(64, activation="relu")(x)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

lstm_model = keras.Model([inp, inm], out)
```

| 항목 | 결과 |
|---|---:|
| 전체 파라미터 | 2,086,275 |
| 테스트 정확도 | 0.608 |
| 가장 낮은 검증 손실 | 0.7561 |
| 해당 Epoch | 2 |

LSTM은 SimpleRNN보다 구조가 복잡하지만 이번 실행의 테스트 정확도는 더 낮았습니다. 복잡한 모델이 언제나 더 좋은 결과를 보장하지는 않는다는 점을 확인할 수 있습니다.

---

## 6. 모델 3: GRU

GRU는 하나의 은닉 상태와 리셋·업데이트 게이트를 사용하는 순환 모델입니다.

![Keras GRU 분류 모델의 구조](assets/lec10_06_gru.png)

```python
inp = keras.Input((max_len,))
inm = keras.Input((max_len,))

x = embed_mask(inp, inm)
x = layers.GRU(units)(x)
x = layers.Dense(64, activation="relu")(x)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

gru_model = keras.Model([inp, inm], out)
```

| 항목 | 결과 |
|---|---:|
| 전체 파라미터 | 2,078,211 |
| 테스트 정확도 | 0.689 |
| 가장 낮은 검증 손실 | 0.7231 |
| 해당 Epoch | 2 |

세 순환 모델 중 GRU가 가장 낮은 검증 손실과 가장 높은 테스트 정확도를 보였습니다.

---

## 7. 같은 조건으로 학습하고 평가하기

![세 순환 모델의 공통 학습 조건과 평가 항목](assets/lec10_07_common_training_evaluation.png)

세 모델에는 같은 학습 설정을 적용합니다.

```python
def fit_model(model, name):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        [X_tr, M_tr],
        y_tr,
        validation_data=([X_val, M_val], y_val),
        batch_size=32,
        epochs=20,
        callbacks=[early_stop],
        verbose=2,
    )
    return history
```

| 설정 | 값 |
|---|---|
| Optimizer | Adam |
| Batch size | 32 |
| 최대 Epoch | 20 |
| Early Stopping 기준 | `val_loss` |
| Patience | 3 |
| 최적 가중치 복원 | 사용 |

Accuracy 하나만으로 모델을 판단하지 않습니다. 학습곡선에서 훈련 손실과 검증 손실의 차이를 확인하고, 혼동행렬에서 어떤 범주를 자주 틀리는지도 함께 살펴봅니다.

---

## 8. 여섯 모델의 테스트 정확도 비교

Lecture 07과 Lecture 10에서 얻은 결과를 하나의 표로 정리합니다.

![여섯 문장 분류 모델의 테스트 정확도 비교](assets/lec10_08_test_accuracy_comparison.png)

| 모델 | 테스트 정확도 |
|---|---:|
| Flatten | 0.811 |
| Masked GAP | 0.856 |
| **Conv1D** | **0.858** |
| SimpleRNN | 0.636 |
| LSTM | 0.608 |
| GRU | 0.689 |

이번 실행에서 관찰한 결과는 다음과 같습니다.

- 전체 여섯 모델 중 Conv1D가 `0.858`로 가장 높은 테스트 정확도를 보였습니다.
- Masked GAP도 `0.856`으로 Conv1D와 매우 가까웠습니다.
- 세 순환 모델 중에는 GRU가 `0.689`로 가장 높았습니다.
- LSTM은 가장 복잡한 순환 구조이지만 이번 데이터에서는 `0.608`로 가장 낮았습니다.

Conv1D와 Masked GAP의 차이는 `0.002`, 즉 0.2%p에 불과합니다. 한 번의 실행만으로 두 모델의 우열을 일반화하기보다는 여러 난수 시드에서 반복해 평균과 변동성을 확인하는 것이 좋습니다.

---

## 9. 검증 손실과 결과 해석

![검증 손실을 이용한 모델 선택과 테스트 데이터의 역할](assets/lec10_09_validation_loss_interpretation.png)

이번 순환 모델들의 검증 손실은 초반에 가장 낮아진 뒤 다시 증가했습니다.

| 순환 모델 | 최저 `val_loss` | Epoch | 테스트 정확도 |
|---|---:|---:|---:|
| SimpleRNN | 0.9327 | 2 | 0.636 |
| LSTM | 0.7561 | 2 | 0.608 |
| **GRU** | **0.7231** | **2** | **0.689** |

```text
훈련 데이터 → 모델 학습
검증 데이터 → 모델과 학습 시점 선택
테스트 데이터 → 선택 완료 후 최종 평가
```

테스트 정확도가 가장 높은 모델을 찾은 뒤 그 모델을 선택하면 테스트 데이터가 모델 선택에 사용됩니다. 따라서 실습 결과 표에서는 테스트 정확도를 비교해 관찰할 수 있지만, 실제 프로젝트에서는 검증 결과로 모델을 선택한 후 테스트 데이터는 한 번만 최종 평가에 사용하는 것이 원칙입니다.

또한 이번 결과만으로 “Conv1D는 항상 순환 모델보다 좋다”거나 “LSTM은 성능이 낮다”고 일반화할 수 없습니다. 짧은 문장, 작은 데이터, 학습 설정, 토크나이저, 난수 초기화 등이 결과에 영향을 줍니다.

---

## 10. 결과를 `res.json`에 저장하기

새로운 세 모델의 결과를 기존 결과 파일에 추가합니다.

```python
path = "res.json"

with open(path, encoding="utf-8") as f:
    old = json.load(f)

old.update(res)

with open(path, "w", encoding="utf-8") as f:
    json.dump(old, f, ensure_ascii=False)

print("저장된 모델:", list(old.keys()))
```

이번 실행에서는 다음 여섯 모델의 결과가 저장되었습니다.

```text
Flatten, GAP, Conv1D, SimpleRNN, LSTM, GRU
```

`res.json`에는 테스트 정확도뿐 아니라 검증 손실 등 후속 비교에 필요한 결과도 함께 저장할 수 있습니다.

---

## 11. 실습 결과를 읽는 방법

이번 실행의 핵심 관찰은 다음과 같습니다.

1. 세 순환 모델 중 GRU의 결과가 가장 좋았습니다.
2. 전체 비교에서는 Conv1D와 Masked GAP가 높은 정확도를 보였습니다.
3. SimpleRNN, LSTM, GRU 모두 검증 손실이 초반에 가장 낮아졌습니다.
4. 구조가 복잡하다고 반드시 테스트 정확도가 높아지는 것은 아닙니다.
5. 한 번의 실행 결과는 확정적인 모델 순위가 아니라 학습 특성을 이해하는 자료입니다.

추가 실험에서는 다음 항목을 확인할 수 있습니다.

- 난수 시드를 바꾸어 3~5회 반복하기
- 각 모델의 평균 정확도와 표준편차 비교하기
- 범주별 Precision, Recall, F1-score 확인하기
- 학습 시간과 파라미터 수를 함께 비교하기
- `units`, Dropout, 학습률을 검증 데이터로 조정하기

---

## 핵심 정리

- 동일한 2,400개 데이터와 고정된 분할로 여섯 모델을 비교했습니다.
- 데이터는 훈련 1,680개, 검증 360개, 테스트 360개로 구성되었습니다.
- SimpleRNN, LSTM, GRU의 테스트 정확도는 각각 `0.636`, `0.608`, `0.689`였습니다.
- 세 순환 모델 중에는 GRU가 가장 높은 테스트 정확도를 보였습니다.
- 전체 모델 중에는 Conv1D가 `0.858`로 가장 높았고, Masked GAP가 `0.856`으로 매우 가까웠습니다.
- 모델은 검증 결과로 선택하고 테스트 데이터는 최종 평가에 사용해야 합니다.
- 이번 수치는 한 번의 실행 결과이므로 반복 실험 없이 일반적인 우열로 해석하지 않습니다.
