# Lecture 04. AI Hub 한국어 병렬 말뭉치 데이터 전처리 실습

이번 실습에서는 한국어 문장을 서브워드 토큰으로 나누고, 딥러닝 모델이 사용할 수 있는 토큰 ID 배열로 변환합니다. 전처리한 데이터는 같은 분할을 유지한 채 이후 MLP, CNN, RNN 실습에서 공통으로 사용합니다.

## 학습 목표

- AI Hub 한국어 대화 문장에서 필요한 범주를 선택할 수 있습니다.
- 서브워드 토크나이저로 문장을 토큰과 토큰 ID로 변환할 수 있습니다.
- 패딩과 잘라내기를 이용해 입력 길이를 맞출 수 있습니다.
- 문자열 라벨을 숫자로 변환할 수 있습니다.
- 데이터를 훈련·검증·테스트 세 부분으로 나눌 수 있습니다.
- 전처리 결과를 NPZ 파일로 저장하고 다시 불러올 수 있습니다.

---

## 1. 오늘 실습의 전체 흐름

이번 실습은 다음 순서로 진행합니다.

![문장에서 전처리 데이터 파일까지의 전체 흐름](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_01_preprocessing_workflow.png)

```text
한국어 문장과 범주
        ↓
3개 범주 선택과 균형 표본 추출
        ↓
서브워드 토큰화
        ↓
패딩과 잘라내기
        ↓
라벨 숫자 변환
        ↓
train / validation / test 분할
        ↓
NPZ 파일 저장
```

Lecture 04에서는 모델을 학습하지 않습니다. 여러 모델이 함께 사용할 **공통 입력 데이터**를 만드는 것이 목표입니다.

---

## 2. 실습 환경 준비하기

Google Colab에서 새 노트북을 열고 필요한 패키지를 설치합니다.

```python
!pip install -q transformers scikit-learn openpyxl
```

이어서 필요한 라이브러리를 불러오고 난수 시드를 고정합니다.

```python
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
```

| 라이브러리 | 주요 역할 |
|---|---|
| NumPy | 배열 처리와 NPZ 파일 저장 |
| pandas | 표 형태의 데이터 처리 |
| Transformers | KLUE-BERT 서브워드 토크나이저 사용 |
| scikit-learn | 라벨 변환과 데이터 분할 |

---

## 3. 데이터 불러오기

실습용으로 정리된 AI Hub 한국어-영어 병렬 말뭉치의 한국어 대화 문장을 불러옵니다.

![엑셀 파일에서 원문과 상황 열을 확인하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_02_load_data.png)

```python
url = "https://raw.githubusercontent.com/hongsukyi/Lectures/main/data/aihub_conver.xlsx"
df = pd.read_excel(url, engine="openpyxl")

print("전체 문장 수:", len(df))
df.head()
```

이번 실습에서 주로 사용하는 열은 다음과 같습니다.

- `원문`: 분류할 한국어 문장
- `상황`: 문장이 속한 원래 범주

> 원자료는 AI Hub에서 제공되지만, 수업에서는 실행 편의를 위해 실습용으로 정리한 엑셀 파일을 사용합니다.

---

## 4. 세 개 범주 선택하기

전체 상황 중에서 `면접`, `학교`, `협상`과 관련된 세 범주만 선택합니다.

![전체 범주에서 면접 학교 협상 범주를 선택하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_03_select_categories.png)

```python
label_map = {
    "취직 면접 상황": "면접",
    "학교생활 (시험, 졸업, 입학 등)": "학교",
    "제안 및 협상하기": "협상",
}

df["label"] = df["상황"].map(label_map)
df = df.dropna(subset=["원문", "label"])

print(df["label"].value_counts())
```

`map()`은 원래의 긴 범주명을 수업에서 이해하기 쉬운 짧은 이름으로 바꿉니다. 선택하지 않은 범주는 결측값이 되므로 `dropna()`로 제외합니다.

---

## 5. 범주별 균형 맞추기

각 범주에서 800문장씩 선택하여 총 2,400문장의 균형 데이터셋을 만듭니다.

![세 범주에서 같은 수의 문장을 선택하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_04_balanced_sampling.png)

```python
N = 800

df = df.groupby("label").sample(n=N, random_state=SEED)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df = df[["원문", "label"]].rename(columns={"원문": "text"})

print("최종 문장 수:", len(df))
print(df["label"].value_counts())
df.head()
```

균형 데이터에서는 세 범주가 각각 800개이므로 Accuracy와 혼동행렬을 비교하기 쉽습니다. `random_state`를 고정하면 다시 실행해도 같은 문장이 선택됩니다.

---

## 6. 서브워드 토크나이저 불러오기

한국어 문장을 단순히 공백으로만 나누지 않고, KLUE-BERT의 서브워드 토크나이저를 사용합니다.

![한국어 문장을 서브워드 토큰으로 나누는 예](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_05_subword_tokenizer.png)

```python
tok = AutoTokenizer.from_pretrained("klue/bert-base")

vocab_size = tok.vocab_size
print("vocab_size:", vocab_size)

sample = df["text"].iloc[0]
print("예문:", sample)
print("토큰:", tok.tokenize(sample))
```

서브워드 토크나이저는 단어를 더 작은 단위로 나눌 수 있습니다. 이번 실습에서 토크나이저는 문장을 분할하고 ID로 변환하는 역할만 합니다. 이후 모델의 임베딩 가중치는 분류 데이터와 함께 새로 학습합니다.

---

## 7. MAX_LEN 정하기

배치로 학습하려면 모든 문장의 입력 길이가 같아야 합니다. 먼저 각 문장의 서브워드 길이 분포를 확인합니다.

![문장 길이 분포에서 MAX_LEN을 정하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_06_choose_max_len.png)

```python
lens = [len(tok.tokenize(t)) for t in df["text"]]

length_95 = int(np.percentile(lens, 95))
max_len = length_95 + 2       # [CLS], [SEP] 자리
max_len = ((max_len + 7) // 8) * 8

print("문장 길이 95%:", length_95)
print("MAX_LEN:", max_len)
```

현재 실습 데이터에서는 다음 값을 사용합니다.

```text
MAX_LEN = 32
```

- 짧은 문장: 뒤에 PAD를 추가합니다.
- 긴 문장: `MAX_LEN` 이후 부분을 잘라냅니다.
- 대부분의 문장을 보존하면서 계산량이 지나치게 커지지 않도록 정합니다.

---

## 8. 문장을 토큰 ID로 바꾸기

토크나이저를 이용해 모든 문장을 고정 길이의 `input_ids`와 `attention_mask`로 변환합니다.

![문장을 토큰 ID와 어텐션 마스크로 변환하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_07_token_ids_mask.png)

```python
out = tok(
    df["text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=max_len,
)

ids = np.array(out["input_ids"])
mask = np.array(out["attention_mask"])

print("ids shape:", ids.shape)
print("예시 ids:", ids[0][:10])
print("예시 mask:", mask[0][:10])
```

결과 배열의 크기는 다음과 같습니다.

```text
input_ids.shape = (2400, 32)
attention_mask.shape = (2400, 32)
```

`input_ids`에는 토큰의 숫자 ID가 들어갑니다. `attention_mask`에서 실제 토큰은 `1`, PAD 위치는 `0`으로 표시됩니다. 여기서 `32`는 임베딩 차원이 아니라 **한 문장에 배정된 토큰 위치 수**입니다.

---

## 9. 라벨을 숫자로 바꾸기

문자열로 된 범주명을 모델이 사용할 수 있는 정수 라벨로 변환합니다.

![문자열 범주를 숫자 라벨로 변환하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_08_label_encoding.png)

```python
le = LabelEncoder()
y = le.fit_transform(df["label"])
classes = le.classes_.tolist()

print("classes:", classes)
print("y[:10]:", y[:10])
```

숫자의 의미는 `classes`의 순서로 확인해야 합니다.

```text
0 → 면접
1 → 학교
2 → 협상
```

숫자의 크기는 범주의 우선순위를 뜻하지 않습니다. 서로 다른 세 범주를 구분하기 위한 코드일 뿐입니다.

---

## 10. train·validation·test 나누기

전체 데이터의 70%는 훈련, 15%는 검증, 15%는 테스트에 사용합니다.

![데이터를 훈련 검증 테스트로 나누는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_09_data_split.png)

```python
idx = np.arange(len(y))

idx_train, idx_tmp = train_test_split(
    idx,
    test_size=0.3,
    random_state=SEED,
    stratify=y,
)

idx_val, idx_test = train_test_split(
    idx_tmp,
    test_size=0.5,
    random_state=SEED,
    stratify=y[idx_tmp],
)

print("train:", len(idx_train))
print("validation:", len(idx_val))
print("test:", len(idx_test))
```

| 데이터 | 비율 | 문장 수 | 역할 |
|---|---:|---:|---|
| Train | 70% | 1,680 | 가중치 학습 |
| Validation | 15% | 360 | 학습 과정 확인과 모델 선택 |
| Test | 15% | 360 | 최종 성능 평가 |

`stratify`를 사용하면 세 부분에서 범주별 비율이 유지됩니다. 이후 모든 모델에서 이 인덱스를 그대로 사용해야 비교가 일관됩니다.

---

## 11. 전처리 결과 저장하기

전처리 결과와 분할 인덱스를 하나의 NPZ 파일로 저장합니다.

![전처리 결과를 저장하고 여러 모델에서 재사용하는 과정](https://cdn.jsdelivr.net/gh/hongsukyi/DL4NLP@main/assets/lec04_10_save_npz.png)

```python
fname = "data.npz"

np.savez(
    fname,
    input_ids=ids,
    attention_mask=mask,
    labels=y,
    texts=df["text"].to_numpy(dtype=object),
    train_idx=idx_train,
    val_idx=idx_val,
    test_idx=idx_test,
    vocab_size=np.int32(vocab_size),
    max_len=np.int32(max_len),
    pad_id=np.int32(tok.pad_token_id),
    num_classes=np.int32(len(classes)),
    class_names=np.array(classes),
)

print("저장 완료:", fname)
```

저장된 내용을 다시 확인합니다.

```python
check = np.load(fname, allow_pickle=True)

print("input_ids shape:", check["input_ids"].shape)
print(
    "train/val/test:",
    len(check["train_idx"]),
    len(check["val_idx"]),
    len(check["test_idx"]),
)
print("class_names:", check["class_names"])
```

다른 Colab 노트북에서 사용할 경우 파일을 내려받습니다.

```python
from google.colab import files

files.download(fname)
```

> **다음 실습:** Lecture 07에서는 이 NPZ 파일을 불러와 Keras로 Flatten, 평균 풀링, Conv1D 모델을 비교합니다.

---

## 핵심 정리

- 서브워드 토크나이저는 문장을 토큰과 토큰 ID로 변환합니다.
- 패딩과 잘라내기로 모든 입력을 `MAX_LEN=32`에 맞춥니다.
- `attention_mask`는 실제 토큰과 PAD 위치를 구분합니다.
- 세 범주를 각각 800문장으로 맞춰 총 2,400문장을 사용합니다.
- 저장된 분할 인덱스를 재사용하면 여러 모델을 같은 조건에서 비교할 수 있습니다.
