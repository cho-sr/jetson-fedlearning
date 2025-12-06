<p align="center">
  <h1>🚀 PROJECT_TITLE</h1>
</p>

<p align="center">
  PROJECT_ONE_LINE_DESCRIPTION
</p>

<p align="center">
  <!-- 필요에 따라 수정 -->
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/Federated-Learning-orange" />
  <img src="https://img.shields.io/badge/Device-%20%7C%20CUDA%20%7C-green" />
</p>

---

## 📚 Table of Contents

- [소개](#-소개)
- [프로젝트 구조](#-프로젝트-구조)
- [환경 및 공통 설정](#-환경-및-공통-설정)
- [모델 구조](#-모델-구조)
- [데이터셋 및 전처리](#-데이터셋-및-전처리)
- [Federated Learning 동작 방식](#-federated-learning-동작-방식)
- [하이퍼파라미터](#-하이퍼파라미터)
- [실행 방법](#-실행-방법)
- [결과 예시](#-결과-예시)
- [To-do](#-to-do)
- [라이선스 / 기타](#-라이선스--기타)

---

## 🧾 소개

이 레포지토리는 **PROJECT_KEYWORDS**를 다루는 예제/연구용 코드입니다.

- 서버 1개 + 클라이언트 N개 구조  
- 각 클라이언트는 **로컬 데이터만 사용**하여 학습하고,  
- 서버는 클라이언트들의 모델을 **PROJECT_AGGREGATION_METHOD (예: FedAvg)** 로 통합하여 글로벌 모델을 갱신합니다.

> 이 템플릿은 Federated Learning / 분산 학습 / Jetson / Edge AI 프로젝트 README에 바로 사용할 수 있도록 구성되어 있습니다.  
> 프로젝트에 맞게 텍스트만 수정해서 사용하세요.

---

## 📁 프로젝트 구조

```text
REPO_NAME/
│
├── server.py          # 서버: 모델 집계, 평가, 로그 출력
├── client1.py         # 클라이언트 1 (예: Non-IID 데이터 A)
├── client2.py         # 클라이언트 2 (예: Non-IID 데이터 B)
│   ...                # 클라이언트가 더 있다면 여기에 추가
│
├── dataset/
│   ├── client1.pt     # Client 1용 데이터셋 (예시)
│   ├── client2.pt     # Client 2용 데이터셋 (예시)
│   └── test.pt        # 서버에서 평가에 사용하는 테스트셋 (예시)
│
├── requirements.txt   # 의존성 패키지 목록
└── README.md
```

🛠 환경 및 공통 설정
필수 환경

Python: 3.10+

주요 라이브러리:

torch (PyTorch 2.x)

torchvision

numpy

tqdm

# 가상환경 생성 & 패키지 설치
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

시드 & 디바이스 설정 예시
SEED = 42

import random, numpy as np, torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"   # Jetson / NVIDIA GPU
else:
    device = "cpu"

print(f"[INFO] Using device: {device}")

🧠 모델 구조

연합학습에 사용되는 Network1은 Jetson 환경을 고려한 경량 CNN입니다.

입력: [B, 3, 192, 192]

출력: [B, 4] (4-class)

구성:

Conv2d → BatchNorm2d → ReLU

Depthwise Separable Conv (DW + PW) 로 파라미터 감소

MaxPool2d로 해상도/연산량 감소

마지막에 AdaptiveAvgPool2d(1×1) + 1×1 Conv classifier

```python
import torch.nn as nn
class Network1(nn.Module):

    def __init__(self, num_classes: int = 4):
        super(Network1, self).__init__()        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)  # [B, C, 1, 1]
        return x.view(x.size(0), -1)
```


🧬 데이터셋 및 전처리
데이터 포맷

.pt 파일 내부:

blob = torch.load("dataset/client1.pt", map_location="cpu")
items = blob["items"]

sample = items[0]
image = sample["tensor"]       # [C, H, W], uint8
label = int(sample["label"])   # 0~3

클래스 라벨
라벨	클래스
0	Glioma
1	Meningioma
2	Notumor
3	Pituitary
Non-IID 분포 예시
Label	Client 1	Client 2
0 Glioma	422개 (14.8%)	899개 (31.5%)
1 Meningioma	1339개 (46.9%)	0개 (0%)
2 Notumor	489개 (17.1%)	1106개 (38.7%)
3 Pituitary	606개 (21.2%)	851개 (29.8%)
Transform 예시
import torch
import torchvision.transforms.v2 as v2

IMG_SIZE = 192

train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

test_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

🔄 Federated Learning 동작 방식

TCP Socket 기반 서버–클라이언트 구조를 사용한다.

1라운드(Round) 흐름

서버 → 각 클라이언트로 글로벌 모델 파라미터 전송

클라이언트는 로컬 데이터셋(client1.pt / client2.pt)으로 local_epochs 만큼 학습

각 클라이언트는 업데이트된 모델 파라미터(state_dict) 를 서버로 전송

서버는 모든 클라이언트의 파라미터를 평균(FedAvg) 하여 새로운 글로벌 모델을 계산

          (1) Broadcast Weights
         ┌─────────────────────┐
         │                     ↓
Server ──┤                Client 1 (local train on client1.pt)
(FedAvg) │                     ↑
         │                     ├── (2) Send updated weights
         │                     ↓
         │                Client 2 (local train on client2.pt)
         │                     ↑
         └─────────────────────┘

⚙ 하이퍼파라미터
서버 (server.py)
IMG_SIZE        = 192
NUM_CLASSES     = 4
DATASET_NAME    = "./dataset/test.pt"

target_accuracy = 90.0   # 목표 정확도 (%)
global_round    = 5      # FL 라운드 수
batch_size      = 64     # 테스트 배치 크기

host            = "127.0.0.1"
port            = 8081

클라이언트 (client1.py / client2.py)
local_epochs = 1
lr           = 0.001
batch_size   = 32

host_ip      = "127.0.0.1"
port         = 8081

# 예시
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss(weight=class_weights)


요약:

항목	값 (예시)
global_round	5
local_epochs	1
batch_size	32(클라) / 64(서버)
lr	0.001
IMG_SIZE	192
🏃 실행 방법
1) 레포지토리 클론 & 환경 세팅
git clone https://github.com/USERNAME/jetson-fed-mri-fl.git
cd jetson-fed-mri-fl

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) 서버 실행
python server.py


※ 필요 시 server.py 내 host, port를 실제 서버 IP에 맞게 수정.

3) 클라이언트 실행 (각 Jetson/터미널에서)
# Client 1
python client1.py

# Client 2
python client2.py


client1.py, client2.py 내 host_ip, port가 서버와 동일해야 함.

📊 결과 예시

(예시: 실제 실험 결과에 맞게 수정 가능)

항목	값
Global Rounds	4
최종 Accuracy	79.71 %
학습 시간	0시간 1분 40.19초
모델 크기	0.1119 MB
추론 시간	2.24 초 (테스트 전체)
Global round [4 / 4] Accuracy : 79.71014492753623 %

학습 소요 시간: 0 시간 1 분 40.19 초
최종 모델 크기: 0.1119 MB

Test: 100%|██████████████████████████████████████████████| 21/21 [00:02<00:00,  9.37it/s]
예측 소요 시간 : 2.24 초
연합학습 종료
