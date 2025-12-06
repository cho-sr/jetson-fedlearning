<p align="center">
<h1>🚀 Jetson Federated Brain MRI Classification</h1>
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
- [주요 특징](#-주요-특징)
- [프로젝트 구조](#-프로젝트-구조)
- [사용 기술](#-사용-기술-tech-stack--techniques)
- [코드에서 사용한 주요 기술](#-코드에서-사용한-주요-기술)
- [모델 구조](#-모델-구조)
- [데이터셋 및 전처리](#-데이터셋-및-전처리)
- [Federated Learning 동작 방식](#-federated-learning-동작-방식)
- [실행 방법](#-실행-방법)
- [결과](#-결과)


---

## 🧾 소개

이 프로젝트는 **Federated Learning 기반 Brain MRI 4-class 분류**를 Jetson/Edge 환경에서도 동작 가능하도록 구현한 코드 레포지토리입니다.  

중앙 서버가 원본 의료 영상을 직접 수집하지 않고,  
각 클라이언트(Jetson/로컬 머신)가 **자신의 로컬 데이터만 사용해 학습**한 뒤  
업데이트된 모델 파라미터만 서버로 보내고, 서버는 이를 **FedAvg(Federated Averaging)** 로 집계하여
글로벌 모델을 갱신하는 구조를 사용합니다.

특히, 이 프로젝트에서는 다음과 같은 목표를 갖고 설계되었습니다.

- 병원/기관 간 **민감한 의료 데이터(raw image)를 공유하지 않는 환경**에서의 학습 시나리오 재현
- **Non-IID 데이터 분포**(클라이언트별 클래스 비율이 크게 다른 상황)를 가정한 현실적인 FL 실험
- Jetson/Edge 디바이스에서도 학습 및 추론이 가능한 **경량 CNN(Network1)** 설계
- 학습 정확도, 추론 시간, 최종 모델 크기 등을 함께 보고하여 **Edge AI 관점의 효율성 평가**


---

## ✨ 주요 특징

- **4-class Brain MRI 분류**
  - 클래스: Glioma, Meningioma, Notumor, Pituitary
  - 입력 해상도: 192×192, 컬러 이미지 3채널

- **Federated Learning 시나리오 구현**
  - 서버 1개 + 클라이언트 N개 구조 (예시: client1, client2)
  - 각 클라이언트는 `.pt` 포맷의 **로컬 데이터셋만 사용**하여 학습
  - 서버는 클라이언트에서 전달받은 `state_dict`를 **FedAvg** 로 평균하여 글로벌 모델 업데이트

- **Non-IID 데이터 분포 실험**
  - Client 1 / Client 2 간 클래스 비율이 크게 다른 **불균형 분포** 설정
  - 한 클라이언트에서는 특정 클래스가 거의 없거나(0%), 다른 클래스가 과다한 상황을 재현

- **경량 CNN(Network1) 설계**
  - Depthwise Separable Convolution(DW + PW) 기반으로 파라미터 및 연산량 감소
  - `AdaptiveAvgPool2d(1×1)` + `1×1 Conv` classifier 구조로 FC layer 제거
  - Jetson / Edge 디바이스에서도 동작 가능한 규모의 모델 크기(~0.04MB대) 달성

- **실험 로그 및 성능 측정**
  - Global round별 테스트 정확도 로그 출력
  - 최종 **정확도, 학습 소요 시간, 모델 파일 크기, 전체 추론 시간** 등을 기록
  - `tqdm` 기반 진행률 표시로 학습 상태 모니터링
 

## 📁 프로젝트 구조

```text
REPO_NAME/
│
├── server.py          # 서버: 모델 집계, 평가, 로그 출력
├── client1.py         # 클라이언트 1 (예: Non-IID 데이터 A)
├── client2.py         # 클라이언트 2 (예: Non-IID 데이터 B)
│
├── dataset/
│   ├── client1.pt     # Client 1용 데이터셋 (예시)
│   ├── client2.pt     # Client 2용 데이터셋 (예시)
│   └── test.pt        # 서버에서 평가에 사용하는 테스트셋 (예시)
│
└── README.md
```

## 🛠 사용 기술 (Tech Stack & Techniques)

### Framework & Libraries
- **Python 3.10+**
- **PyTorch 2.x**
  - `nn.Module` 기반 커스텀 경량 CNN(Network1) 구현
  - `CrossEntropyLoss` 및 클래스 가중치(class weights)를 활용한 불균형 데이터 대응
- **torchvision**
  - `transforms.v2`를 사용한 Resize, Normalize, Data Augmentation(RandomHorizontalFlip 등)
- **numpy, random**
  - 시드 고정 및 재현성(Reproducibility) 확보
- **tqdm**
  - 학습/테스트 진행 상황을 시각적으로 표시

### Federated Learning & Training Logic
- **FedAvg(Federated Averaging)**
  - 클라이언트별 학습 후 `state_dict`를 서버에서 수신
  - 파라미터를 단순 평균하여 새로운 글로벌 모델 생성
- **Non-IID 설정**
  - Client 1 / Client 2 간 클래스 분포를 인위적으로 다르게 구성
  - 의료 데이터 환경에서 자주 발생하는 **데이터 불균형 & 기관 간 분포 차이**를 실험적으로 반영
- **하이퍼파라미터 튜닝**
  - `global_round`, `local_epochs`, `batch_size`, `lr` 등을 조정하며 성능/시간/자원 사용량 사이의 트레이드오프 확인
  - 목표 정확도(`target_accuracy`)를 기준으로 한 조기 종료(Early stopping) 시나리오도 고려

### 시스템 & 엔지니어링
- **TCP Socket 기반 통신**
  - `socket`, `struct`, `pickle` 등을 사용해 서버–클라이언트 간 모델 파라미터 송수신
  - 단일 머신 로컬(loopback) 뿐 아니라, IP/Port 설정 변경만으로 LAN 환경으로 확장 가능
- **디바이스 대응**
  - Apple Silicon 환경에서는 **MPS(Metal Performance Shaders)** 사용
  - NVIDIA / Jetson 환경에서는 **CUDA** 사용
  - GPU/MPS가 없는 경우 자동으로 **CPU**로 폴백
- **재현성 & 안정성**
  - `random`, `numpy`, `torch`, `torch.cuda` 시드 고정
  - `torch.backends.cudnn.deterministic = True` 설정으로 일관된 결과 재현


## 🧩 코드에서 사용한 주요 기술

### 1. 경량 CNN 모델(Network1) 설계

- `nn.Module` 기반 **커스텀 경량 CNN** 구현
- 구조 특징:
  - 입력: `[B, 3, 192, 192]`
  - 출력: `[B, 4]` (Brain MRI 4-class 분류)
  - **Depthwise Separable Convolution**
    - `groups=in_channels`인 `Conv2d`(DW) + `1x1 Conv2d`(PW) 조합  
    - 채널 수: `3→20→30→45→67→100` 단계적으로 증가
  - `MaxPool2d`로 해상도와 연산량 감소
  - `ReLU6` 활성화 사용 (모바일/경량 환경에서 자주 쓰이는 활성함수)
  - 마지막에 `AdaptiveAvgPool2d(1x1)` + `Conv2d(1x1)`  
    → FC 레이어 없이 **채널 축만 남기는 1×1 Conv classifier**
- 분류기(`self.classifier`):
  - `BatchNorm2d(100)` + `Dropout(0.1)` + `Conv2d(100, num_classes, kernel_size=1)`  
  → 경량 구조 유지하면서도 **정규화 + 규제(regularization)** 반영

---

### 2. 데이터셋 로딩 & 전처리 파이프라인

- `.pt` 파일 구조:
  - `blob = torch.load(pt_path, map_location="cpu", weights_only=False)`
  - `blob["items"]` 안에 `{ "tensor": 이미지, "label": 라벨 }` 형태의 리스트 저장
- `CustomDataset` 구현:
  - `self.images = [item["tensor"] for item in blob["items"]]`
  - `self.labels = [int(item["label"]) for item in blob["items"]]`
  - `__getitem__`에서:
    - `x = self.images[idx].float() / 255.0` 로 0~1 스케일링
    - 필요 시 `transform(x)` 적용
- 전처리(Transform):
  - **Train (client1)**:
    - `Resize((IMG_SIZE, IMG_SIZE))`
    - `RandomHorizontalFlip(0.5)` (간단한 데이터 증강)
    - `ToDtype(torch.float32, scale=True)`
    - `Normalize(mean, std)` (ImageNet 통계 기반)
  - **Test (server)**:
    - 동일한 Resize & Normalize, 증강 없이 평가용으로만 사용
- `DataLoader`:
  - 클라이언트: `batch_size=32`, `shuffle=True`
  - 서버(테스트): `batch_size=64`, `shuffle=False`

---

### 3. 학습 로직 & 최적화 기법

#### (1) Mixed Precision Training (AMP) – 클라이언트

- `torch.amp.GradScaler('cuda')` 및 `torch.amp.autocast('cuda', enabled=use_amp)` 사용
- `device == "cuda"` 인 경우에만 **AMP 활성화**
  - Forward & Loss 계산을 autocast 영역에서 수행
  - `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`로 학습
- CUDA가 아닐 경우에는 **일반 FP32 학습**으로 자동 폴백

#### (2) Optimizer & Loss

- Optimizer:
  - `optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)`
  - weight decay로 **L2 규제**를 적용해 과적합 완화
- Loss:
  - `CrossEntropyLoss(weight=class_weights)`
  - `class_weights = [3.1, 4.0, 2.5, 3.3]`  
    → **클래스 불균형(class imbalance)** 보정을 위한 클래스별 가중치 적용

#### (3) 재현성(Reproducibility)

- 고정 시드:
  - `SEED = 42`
  - `random`, `numpy`, `torch`, `torch.cuda` 각각 seed 설정
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`  
  → 연산 최적화 대신 **결과 일관성**을 우선

---

### 4. 디바이스 선택 & Half Precision Inference

- 클라이언트(main):
  - `mps` → `cuda` → `cpu` 순으로 사용 가능 디바이스 자동 선택
- 서버:
  - 시작 시 device를 전역으로 설정 (`mps` / `cuda` / `cpu`)
- Inference 최적화:
  - 서버 측 `measure_accuracy`:
    - `model = Network1().to(device); model.load_state_dict(global_model)`
    - `model.half()` → 파라미터를 **FP16**으로 변환
    - 입력도 `inputs.to(device).half()`로 FP16 변환
  - 추론 시간을 측정하여 **예측 소요 시간(inference_time)** 로그 출력

---

### 5. Federated Learning 통신 구조 (TCP Socket)

#### (1) 클라이언트(client1.py)

- 서버와 TCP 연결:
  - `client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)`
  - `client.connect((host_ip, port))`
- 서버로부터 글로벌 모델 수신:
  - 먼저 4바이트 길이 정보 수신: `data_size = struct.unpack('>I', client.recv(4))[0]`
  - 그 길이만큼 payload 수신 후 `pickle.loads`로 역직렬화
  - `OrderedDict`로 감싸서 `model.load_state_dict(weight, strict=True)`
- 로컬 학습:
  - `train(model, criterion, optimizer, train_loader)` 호출
- 로컬 모델 전송:
  - `model.state_dict().items()`를 dict로 감싸 `pickle.dumps`
  - `struct.pack('>I', len(model_data))`로 길이 먼저 전송 후, 본문 전송
- `select.select`로 서버 종료 신호 감지 후 "Federated Learning finished" 출력

#### (2) 서버(server.py)

- 멀티스레드 서버:
  - `server.listen()` 후, 두 개의 클라이언트 연결 수락
  - `handle_client`를 스레드로 실행 (connection1, connection2)
- 초기 브로드캐스트:
  - 각 클라이언트에 초기 `model.state_dict()` 전송
- 라운드별 처리:
  - 각 스레드에서 클라이언트로부터 로컬 모델 수신 → `model_list`에 저장
  - 두 모델이 모두 도착하면:
    - `average_models(model_list)`로 **FedAvg** 수행
    - `measure_accuracy`로 테스트셋 정확도 측정
    - `current_round` 증가 및 로그 출력
    - `get_model_size`로 최종 모델 크기(MB 단위) 계산
- 동기화:
  - `threading.Semaphore`를 활용해 두 클라이언트 스레드가  
    **동시에 한 라운드를 마치고 다음 라운드로 넘어가도록** 제어
- 종료 조건:
  - `current_round == global_round` 또는  
    `global_accuracy >= target_accuracy` 만족 시:
    - 최종 글로벌 모델을 한 번 더 전송 후 커넥션 종료

---

### 6. 실험 지표 계산 및 로깅

- 학습 시간 측정:
  - `training_start = time.time()` ~ `training_end = time.time()`  
  - 시/분/초 단위로 포맷팅해서 **총 학습 소요 시간** 출력
- 모델 크기 측정:
  - `pickle.dumps(dict(global_model.state_dict().items()))` 길이를  
    `MB` 단위로 환산하여 출력
- 최종 결과 로그:
  - `학습 성능 : {global_accuracy} %`
  - `학습 소요 시간: H 시간 M 분 S 초`
  - `최종 모델 크기: X.XXXX MB`
  - `예측 소요 시간 : T 초`
  - `"연합학습 종료"` 메시지로 종료 시점 명확히 표시


## 🧠 모델 구조

연합학습에 사용되는 Network1은 Jetson 환경을 고려한 경량 CNN입니다.

- 입력: [B, 3, 192, 192]

- 출력: [B, 4] (4-class)

- 구성:

  - Conv2d → BatchNorm2d → ReLU
  - Depthwise Separable Conv (DW + PW) 로 파라미터 감소
  - MaxPool2d로 해상도/연산량 감소
  - 마지막에 AdaptiveAvgPool2d(1×1) + 1×1 Conv classifier

```python
class Network1(nn.Module):
    def __init__(self, num_classes=4):
        super(Network1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU6(inplace=True),

            nn.Conv2d(20, 20, 3, padding=1, groups=20, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU6(inplace=True),
            nn.Conv2d(20, 30, 1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(30, 30, 3, padding=1, groups=30, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU6(inplace=True),
            nn.Conv2d(30, 45, 1, bias=False),
            nn.BatchNorm2d(45),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(45, 45, 3, padding=1, groups=45, bias=False),
            nn.BatchNorm2d(45),
            nn.ReLU6(inplace=True),
            nn.Conv2d(45, 67, 1, bias=False),
            nn.BatchNorm2d(67),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(67, 67, 3, padding=1, groups=67, bias=False),
            nn.BatchNorm2d(67),
            nn.ReLU6(inplace=True),
            nn.Conv2d(67, 100, 1, bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(100),
            nn.Dropout(0.1),
            nn.Conv2d(100, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
        return x.view(x.size(0), -1)
```


## 🧬 데이터셋 및 전처리
데이터 포맷

.pt 파일 내부:

blob = torch.load("dataset/client1.pt", map_location="cpu")
items = blob["items"]

sample = items[0]
image = sample["tensor"]       # [C, H, W], uint8
label = int(sample["label"])   # 0~3

클래스 라벨

Non-IID 분포
```
Label	        Client 1	     Client 2
0 Glioma	    422개 (14.8%)	 899개 (31.5%)
1 Meningioma	1339개 (46.9%)	 0개 (0%)
2 Notumor	    489개 (17.1%)	 1106개 (38.7%)
3 Pituitary     606개 (21.2%)	 851개 (29.8%)
```
```python
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
```
## 🔄 Federated Learning 동작 방식

TCP Socket 기반 서버–클라이언트 구조를 사용한다.

1라운드(Round) 흐름

서버 → 각 클라이언트로 글로벌 모델 파라미터 전송

클라이언트는 로컬 데이터셋(client1.pt / client2.pt)으로 local_epochs 만큼 학습

각 클라이언트는 업데이트된 모델 파라미터(state_dict) 를 서버로 전송

서버는 모든 클라이언트의 파라미터를 평균(FedAvg) 하여 새로운 글로벌 모델을 계산
```
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
```


## 🏃 실행 방법
### 1) 레포지토리 클론 & 환경 세팅
```
git clone https://github.com/cho-sr/jetson-fedlearning.git
cd jetson-fedlearning

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 서버 실행
```python
python server.py
```
※ 필요 시 server.py 내 host, port를 실제 서버 IP에 맞게 수정.

### 3) 클라이언트 실행 (각 Jetson/터미널에서)
- Client 1
  ```python
  python client1.py
- Client 2
  ```python
  python client2.py

client1.py, client2.py 내 host_ip, port가 서버와 동일해야 함.

## 📊 결과


- 학습 성능 : 81.46453089244851 %

- 학습 소요 시간: 0 시간 0 분 16.54 초

- 최종 모델 크기: 0.0491 MB

- 예측 소요 시간 : 0.85 초
