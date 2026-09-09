# Lecture 07. Keras로 비교하는 MLP와 1D CNN 실습

이번 실습에서는 Lecture 04에서 저장한 동일한 데이터를 이용해 세 가지 문장 분류 모델을 만듭니다. TensorFlow/Keras를 기본 프레임워크로 사용하며, Flatten, PAD를 제외한 평균 풀링, Conv1D가 문장을 처리하는 방식의 차이를 살펴봅니다.

## 학습 목표

- 저장된 NPZ 파일에서 입력과 데이터 분할을 불러올 수 있습니다.
- Keras Embedding층으로 토큰 ID를 벡터로 변환할 수 있습니다.
- `attention_mask`를 이용해 PAD 위치를 처리할 수 있습니다.
- Flatten, Masked GAP, Conv1D 모델을 구현할 수 있습니다.
- 학습곡선, Accuracy, 혼동행렬을 해석할 수 있습니다.
- 분류 직전 특징을 t-SNE로 시각화할 수 있습니다.

> 이 장에서는 Keras 코드를 기본으로 사용합니다. PyTorch 버전은 별도의 링크로 제공합니다.

---

## 1. 세 모델 비교 실습

세 모델은 같은 데이터와 같은 훈련·검증·테스트 분할을 사용합니다.

![같은 데이터로 Flatten GAP Conv1D를 비교하는 흐름](assets/lec07_01_three_models.png)

| 모델 | 문장을 요약하는 방식 |
|---|---|
| MLP + Flatten | 모든 위치의 임베딩을 순서대로 펼침 |
| MLP + Masked GAP | PAD를 제외한 토큰 임베딩의 평균 |
| Conv1D + Global Max Pooling | 인접한 토큰 구간에서 강한 특징을 선택 |

같은 데이터와 학습 설정을 사용하지만 모델별 파라미터 수는 다릅니다. 따라서 이 실습은 엄밀한 성능 순위 결정이 아니라 **처리 방식에 따른 차이를 관찰하는 실습**입니다.

---

## 2. 실습 환경 준비하기

새로운 Google Colab 런타임에서는 한글 그래프를 위한 패키지를 먼저 설치합니다.

```python
!pip install -q koreanize-matplotlib
```

필요한 라이브러리를 불러오고 난수 시드를 고정합니다.

```python
import json
import os

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

---

## 3. 저장된 데이터와 분할 불러오기

Lecture 04에서 만든 다음 파일을 Colab 작업 폴더에 업로드합니다.

```text
data.npz
```

![NPZ 파일에서 입력과 데이터 분할을 불러오는 과정](assets/lec07_02_load_npz.png)

```python
from google.colab import files

uploaded = files.upload()
```

```python
data = np.load(
    "data.npz",
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

예상되는 데이터 수는 `1,680 / 360 / 360`입니다.

---

## 4. 토큰 ID를 임베딩 벡터로 바꾸기

Embedding층은 각 토큰 ID를 64개의 실수로 이루어진 벡터로 변환합니다.

![토큰 ID를 64차원 임베딩 벡터로 변환하는 과정](assets/lec07_03_embedding.png)

```text
토큰 ID 배열: [batch, 32]
        ↓ Embedding
임베딩 배열: [batch, 32, 64]
```

이번 실습에서는 사전학습 BERT 모델을 불러오는 것이 아닙니다. KLUE-BERT 토크나이저의 ID 체계를 사용하지만, Keras Embedding층의 값은 분류 모델과 함께 처음부터 학습합니다.

```python
dim = 64

def embed(input_ids, attention_mask):
    x = layers.Embedding(vocab_size, dim)(input_ids)
    mask_float = layers.Lambda(
        lambda m: tf.cast(m, tf.float32)[..., None]
    )(attention_mask)
    return layers.Multiply()([x, mask_float])
```

---

## 5. PAD를 실제 내용과 구분하기

PAD는 입력 길이를 맞추기 위한 자리이므로 문장의 실제 내용처럼 처리하면 안 됩니다.

![어텐션 마스크로 실제 토큰과 PAD를 구분하는 과정](assets/lec07_04_padding_mask.png)

- Embedding 출력에서 PAD 위치를 `0`으로 만듭니다.
- 평균 풀링에서는 실제 토큰 수로만 나눕니다.
- 글로벌 맥스 풀링에서는 PAD 위치가 최댓값으로 선택되지 않게 합니다.

```text
attention_mask = 1 → 실제 토큰
attention_mask = 0 → PAD
```

---

## 6. 같은 설정으로 학습하기

세 모델에 같은 옵티마이저, 배치 크기, 최대 Epoch를 적용합니다.

![세 모델에 공통 학습 설정을 적용하는 과정](assets/lec07_05_training_settings.png)

```python
def fit_model(model):
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

Early Stopping은 검증 손실이 더 이상 좋아지지 않을 때 학습을 멈춥니다. `restore_best_weights=True`는 검증 손실이 가장 작았던 시점의 가중치를 복원합니다.

---

## 7. 모델 1: Flatten으로 모든 값 펼치기

Flatten은 `32 × 64` 형태의 임베딩을 하나의 긴 벡터로 펼칩니다.

![임베딩 행렬을 펼쳐 MLP로 분류하는 과정](assets/lec07_06_flatten_model.png)

```python
inp = keras.Input(
    shape=(max_len,), dtype="int32", name="input_ids"
)
inm = keras.Input(
    shape=(max_len,), dtype="int32", name="attention_mask"
)

x = embed(inp, inm)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

flat_model = keras.Model([inp, inm], out)
flat_model.summary()
```

```python
h_flat = fit_model(flat_model)
```

Flatten은 모든 위치의 값을 보존하지만 입력 길이가 길어지면 파라미터 수가 크게 늘어납니다.

---

## 8. 모델 2: PAD를 제외한 평균 풀링

두 번째 모델은 실제 토큰의 임베딩을 더한 뒤 유효한 토큰 수로 나눕니다.

![PAD를 제외하고 평균 문장 벡터를 만드는 과정](assets/lec07_07_masked_gap.png)

```python
inp = keras.Input(
    shape=(max_len,), dtype="int32", name="input_ids"
)
inm = keras.Input(
    shape=(max_len,), dtype="int32", name="attention_mask"
)

x = embed(inp, inm)

token_sum = layers.Lambda(
    lambda t: tf.reduce_sum(t, axis=1)
)(x)

token_count = layers.Lambda(
    lambda m: tf.maximum(
        tf.reduce_sum(tf.cast(m, tf.float32), axis=1, keepdims=True),
        1.0,
    )
)(inm)

avg = layers.Lambda(
    lambda t: t[0] / t[1]
)([token_sum, token_count])

x = layers.Dense(64, activation="relu")(avg)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

gap_model = keras.Model([inp, inm], out)
gap_model.summary()
```

```python
h_gap = fit_model(gap_model)
```

일반적인 GAP와 비슷하지만 PAD를 제외하고 평균을 계산하므로 **Masked GAP**라고 부를 수 있습니다. 문장을 64차원 벡터 하나로 요약하기 때문에 Flatten보다 파라미터 수가 적습니다.

---

## 9. 모델 3: Conv1D로 이웃한 토큰 살펴보기

Conv1D 모델은 커널 크기 3으로 서로 이웃한 토큰의 지역 패턴을 찾습니다.

![Conv1D와 글로벌 맥스 풀링으로 문장을 분류하는 과정](assets/lec07_08_conv1d_model.png)

```python
inp = keras.Input(
    shape=(max_len,), dtype="int32", name="input_ids"
)
inm = keras.Input(
    shape=(max_len,), dtype="int32", name="attention_mask"
)

x = embed(inp, inm)
x = layers.Conv1D(
    filters=64,
    kernel_size=3,
    padding="same",
    activation="relu",
)(x)

# PAD 위치가 글로벌 맥스 풀링에서 선택되지 않게 합니다.
pad_bias = layers.Lambda(
    lambda m: (1.0 - tf.cast(m, tf.float32))[..., None] * -1e9
)(inm)
x = layers.Add()([x, pad_bias])
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(64, activation="relu")(x)
feat = layers.Dense(16, activation="relu", name="feat")(x)
out = layers.Dense(n_cls, activation="softmax")(feat)

conv_model = keras.Model([inp, inm], out)
conv_model.summary()
```

```python
h_conv = fit_model(conv_model)
```

Global Max Pooling은 각 필터가 문장 전체에서 만든 값 중 가장 큰 값 하나를 남깁니다. 즉, 각 지역 특징이 문장의 어느 위치에서 가장 강하게 나타났는지를 요약합니다.

---

## 10. 학습곡선 살펴보기

훈련과 검증 데이터의 Loss와 Accuracy를 함께 그립니다.

![훈련과 검증의 Loss와 Accuracy 학습곡선](assets/lec07_09_learning_curves.png)

```python
def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="validation")
    ax[0].set_title(f"{name} - Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="validation")
    ax[1].set_title(f"{name} - Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
```

```python
plot_history(h_flat, "Flatten")
plot_history(h_gap, "Masked GAP")
plot_history(h_conv, "Conv1D")
```

- 훈련 Loss는 감소하지만 검증 Loss가 증가하면 과적합 가능성이 있습니다.
- 훈련과 검증 Accuracy의 차이가 지나치게 크지 않은지 확인합니다.
- Epoch 수만 보지 말고 검증 Loss가 가장 좋았던 지점을 확인합니다.

---

## 11. 테스트 정확도와 혼동행렬

테스트 문장마다 확률이 가장 높은 범주를 예측값으로 선택합니다.

![세 모델의 테스트 정확도와 혼동행렬 비교](assets/lec07_10_test_confusion_matrix.png)

```python
def evaluate_model(model, name):
    prob = model.predict([X_te, M_te], verbose=0)
    pred = prob.argmax(axis=1)
    acc = (pred == y_te).mean()

    print(f"[{name}] Test Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(n_cls), classes)
    plt.yticks(range(n_cls), classes)

    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("예측")
    plt.ylabel("정답")
    plt.tight_layout()
    plt.show()

    return pred, float(acc)
```

```python
pred_flat, acc_flat = evaluate_model(flat_model, "Flatten")
pred_gap, acc_gap = evaluate_model(gap_model, "Masked GAP")
pred_conv, acc_conv = evaluate_model(conv_model, "Conv1D")
```

혼동행렬에서는 대각선이 정답을 맞힌 수이고, 대각선 밖의 값은 서로 혼동한 수입니다. 세 범주의 데이터 수가 같더라도 Accuracy만 보지 말고 어떤 범주를 자주 혼동하는지 함께 확인합니다.

---

## 12. t-SNE로 16차원 특징 살펴보기

각 모델의 `feat`층은 분류 직전에 문장을 16차원 벡터로 표현합니다. t-SNE를 이용하면 이 벡터를 2차원에 배치해 살펴볼 수 있습니다.

![16차원 문장 특징을 t-SNE로 시각화하는 과정](assets/lec07_11_tsne.png)

```python
def plot_tsne(model, name):
    feat_model = keras.Model(
        model.input,
        model.get_layer("feat").output,
    )

    feat = feat_model.predict([X_te, M_te], verbose=0)

    xy = TSNE(
        n_components=2,
        random_state=SEED,
        init="pca",
        perplexity=30,
        n_jobs=1,
    ).fit_transform(feat)

    plt.figure(figsize=(5, 5))
    for c in range(n_cls):
        selected = y_te == c
        plt.scatter(
            xy[selected, 0],
            xy[selected, 1],
            s=12,
            label=classes[c],
        )

    plt.title(f"{name} - t-SNE")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

```python
plot_tsne(flat_model, "Flatten")
plot_tsne(gap_model, "Masked GAP")
plot_tsne(conv_model, "Conv1D")
```

> t-SNE는 고차원 특징을 탐색하기 위한 그림입니다. 점들이 분리되어 보인다는 이유만으로 모델 성능이 좋다고 결론 내리면 안 됩니다. 최종 평가는 테스트 지표와 혼동행렬을 함께 사용합니다.

---

## 13. 결과 저장하기

다음 RNN 비교 실습(Lab3)에서 이어서 사용할 수 있도록, Lab2·Lab3와 같은 방식으로 결과를 저장합니다. 모델 이름을 키로 하는 딕셔너리에 각 모델의 결과를 담고, 기존 `res.json`이 있으면 이어서 누적합니다.

![세 모델의 결과를 JSON 파일로 저장하는 과정](assets/lec07_12_save_results.png)

```python
res = {
    "Flatten": {
        "loss": h_flat.history["loss"],
        "val_loss": h_flat.history["val_loss"],
        "acc": h_flat.history["accuracy"],
        "val_acc": h_flat.history["val_accuracy"],
        "pred": pred_flat.tolist(),
        "test_acc": acc_flat,
    },
    "GAP": {
        "loss": h_gap.history["loss"],
        "val_loss": h_gap.history["val_loss"],
        "acc": h_gap.history["accuracy"],
        "val_acc": h_gap.history["val_accuracy"],
        "pred": pred_gap.tolist(),
        "test_acc": acc_gap,
    },
    "Conv1D": {
        "loss": h_conv.history["loss"],
        "val_loss": h_conv.history["val_loss"],
        "acc": h_conv.history["accuracy"],
        "val_acc": h_conv.history["val_accuracy"],
        "pred": pred_conv.tolist(),
        "test_acc": acc_conv,
    },
}

path = "res.json"
old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
old.update(res)
json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False)

print("저장된 모델:", list(old.keys()))
```

Colab 런타임이 바뀌기 전에 결과 파일을 내려받습니다.

```python
from google.colab import files

files.download("res.json")
```

---

## 핵심 정리

- Keras Embedding층은 토큰 ID를 학습 가능한 64차원 벡터로 바꿉니다.
- Flatten은 모든 위치의 값을 펼치고, Masked GAP는 PAD를 제외한 평균을 사용합니다.
- Conv1D는 인접한 토큰 구간의 특징을 찾고 글로벌 맥스 풀링으로 요약합니다.
- 학습곡선, 테스트 Accuracy, 혼동행렬은 서로 다른 정보를 제공합니다.
- t-SNE는 특징을 탐색하는 도구이며 성능 지표를 대신하지 않습니다.
- 동일한 데이터 분할을 재사용해야 모델 비교가 일관됩니다.

