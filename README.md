<h1 align="center">🚀 Jetson Federated Brain MRI Classification</h1>


<p align="center">
  Federated Learning 기반 경량 CNN으로 Brain MRI 4-class를 분류하는 Jetson/Edge AI 실험 프로젝트
</p>

<p align="center">
  <!-- 필요에 따라 수정 -->
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/Federated-Learning-orange" />
  <img src="https://img.shields.io/badge/Device-%20%7C%20CUDA%20%7C-green" />
</p>

<br>


---

## 📚 Table of Contents

- [소개](#-소개)
- [주요 특징](#-주요-특징)
- [프로젝트 구조](#-프로젝트-구조)
- [사용 기술](#-사용-기술-tech-stack--techniques)
- [실행 방법](#-실행-방법)
- [결과](#-결과)

<br>

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

<br>

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
  - Jetson / Edge 디바이스에서도 동작 가능한 규모의 모델 크기(~0.11MB대) 달성

- **실험 로그 및 성능 측정**
  - Global round별 테스트 정확도 로그 출력
  - 최종 **정확도, 학습 소요 시간, 모델 파일 크기, 전체 추론 시간** 등을 기록
  - `tqdm` 기반 진행률 표시로 학습 상태 모니터링
 
<br>

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

<br>

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
---

### 1. 모델링 & 아키텍처

이 프로젝트에서는 Jetson/Edge 환경에서도 동작 가능한 **경량 CNN(Network1)** 을 직접 설계하여 사용했습니다.  
기본 아이디어는 *연산량과 파라미터 수를 줄이면서도* 192×192 해상도의 Brain MRI 이미지를 4개 클래스로 안정적으로 분류하는 것입니다.

- **Depthwise Separable Convolution 기반 경량 구조**
  - 일반적인 Conv2d 대신, `groups=in_channels`인 **Depthwise Conv**와 `1×1 Pointwise Conv`를 조합하여 사용했습니다.
  - 이런 구조는 MobileNet 계열에서 사용하는 방식으로,  
    채널별로 먼저 공간 연산(DW)을 하고, 이후 채널 방향으로만 합치는 연산(PW)을 수행해 **파라미터와 FLOPs를 크게 줄이는 효과**가 있습니다.
  - 채널 수는 `3 → 16 → 32 → 64 → 128 → 256`으로 점진적으로 증가시켜,  
    초반에는 연산량을 아끼고 뒤로 갈수록 표현력을 확보하도록 설계했습니다.

- **ReLU6 활성함수**
  - 모든 블록에 `ReLU6`를 사용했습니다.  
  - ReLU6는 출력 상한을 6으로 제한하는 ReLU 변형으로, 모바일/경량 환경에서 양자화나 고정소수점 연산에 안정적인 활성함수로 자주 사용됩니다.

- **해상도 감소 & 특징 추출**
  - 중간마다 `MaxPool2d`를 배치해 공간 해상도를 단계적으로 줄였습니다.
  - 이를 통해 연산량을 줄이는 동시에, 점점 더 추상적인 high-level feature를 학습하도록 유도했습니다.

- **Fully-connected 없이 1×1 Conv로 분류**
  - 마지막에는 `AdaptiveAvgPool2d(1×1)`으로 H×W를 1×1로 줄여 채널 축만 남게 하고,
  - `Conv2d(256, num_classes, kernel_size=1)`을 적용하여 **완전연결층(FC layer) 없이** 바로 클래스 로짓을 뽑습니다.
  - 이 구조는 파라미터 수를 최소화하면서도 CNN의 장점을 유지할 수 있어, Edge 디바이스에서 특히 효율적입니다.

---

### 2. Federated Learning & 통신 구조

이 프로젝트의 핵심은 **Federated Learning(FedAvg)** 을 실제 동작 가능한 형태로 구현했다는 점입니다.  
데이터는 각 클라이언트에만 존재하고, 서버는 오직 **모델 파라미터(state_dict)** 만 주고받습니다.

- **FedAvg 기반 글로벌 모델 업데이트**
  - 각 클라이언트는 자신의 로컬 데이터(`client1.pt`, `client2.pt`)로 몇 epoch 학습한 후,
    `model.state_dict()`를 서버로 전송합니다.
  - 서버는 두 클라이언트의 `state_dict`를 받아 같은 key끼리 평균을 내는 방식으로 **FedAvg** 를 직접 구현합니다.
  - 이렇게 생성된 평균 파라미터가 새로운 **글로벌 모델**이 되어 다음 라운드에 다시 브로드캐스트됩니다.

- **TCP Socket 기반 서버–클라이언트 통신**
  - Python의 `socket` 모듈을 사용해 **TCP 통신**을 구현했습니다.
  - 모델 파라미터는 `pickle`로 직렬화하고, `struct`를 이용해 **전송할 바이트 길이를 먼저 4바이트로 보내고, 그 다음 payload를 전송**하는 방식으로 구현했습니다.
    - 수신 측에서는 먼저 길이(4바이트)를 읽고, 그 길이만큼 정확히 다시 읽어 전체 payload를 복원합니다.
  - 이런 형태는 실제 네트워크 환경에서도 자주 쓰이는 **길이 프레임 기반 프로토콜 패턴**을 따릅니다.

- **2개 클라이언트 스레드 & 동기화**
  - 서버는 `threading.Thread`를 활용해 **클라이언트별로 하나의 스레드**를 할당합니다.
  - 각 스레드는:
    1. 글로벌 모델 전송  
    2. 로컬 모델 수신  
    3. 서버의 FedAvg 결과를 다시 전송  
    이 과정을 라운드마다 반복합니다.
  - `threading.Semaphore`를 이용해, **두 클라이언트의 모델이 모두 도착한 다음에만** FedAvg를 수행하도록 동기화하여,
    라운드별로 일관된 업데이트가 이루어지도록 했습니다.
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

---

### 3. 데이터셋 & 전처리

Federated Learning의 현실적인 시나리오를 만들기 위해 **Non-IID 분포의 Brain MRI 데이터셋**을 사용하고,  
이를 `.pt` 포맷으로 저장한 뒤 커스텀 `Dataset` 클래스로 로딩하는 방식을 채택했습니다.

- **.pt 포맷 커스텀 Dataset**
  - `torch.load(pt_path)`로 불러오는 `.pt` 파일 내부에는 `blob["items"]` 리스트가 있고,
    각 요소는 `{"tensor": 이미지 텐서, "label": 라벨 정수}` 형태의 딕셔너리로 구성되어 있습니다.
  - `CustomDataset`에서는 이 리스트를 그대로 `self.images`, `self.labels`로 나눠 담고,
    `__getitem__`에서 `x.float() / 255.0`으로 0~1 스케일링 후, 필요 시 `transform`을 적용합니다.
  - 이 방식은 **PyTorch Dataset 인터페이스를 그대로 사용하면서도, 미리 전처리된 텐서 데이터를 손쉽게 재사용**할 수 있게 해 줍니다.

- **클라이언트별 Non-IID 분포**
  - Client1과 Client2는 클래스 분포가 크게 다르게 설계되어 있습니다.
    - 예: Client2에는 Meningioma(라벨 1)가 0개인 반면, Client1에는 해당 라벨이 다수 존재
  - 이를 통해 “특정 병원에는 특정 병변이 거의 없다” 같은 **현실적인 의료 데이터 편향 상황**을 모사했습니다.
  
Non-IID 분포
```
Label	        Client 1	     Client 2
0 Glioma	    422개 (14.8%)	 899개 (31.5%)
1 Meningioma	1339개 (46.9%)	 0개 (0%)
2 Notumor	    489개 (17.1%)	 1106개 (38.7%)
3 Pituitary     606개 (21.2%)	 851개 (29.8%)
```

- **전처리 & 간단한 데이터 증강**
  - `torchvision.transforms.v2`를 이용하여 다음과 같은 파이프라인을 구성했습니다.
    - `Resize((IMG_SIZE, IMG_SIZE))` : 모든 이미지를 192×192로 통일
    - `RandomHorizontalFlip(0.5)` : 간단한 좌우 반전 증강으로 데이터 다양성 확보
    - `ToDtype(torch.float32, scale=True)` : float32 변환 및 0~1 스케일링
    - `Normalize(mean, std)` : ImageNet 통계 기반 정규화
  - 서버의 테스트셋은 **동일한 Resize/Normalize**를 사용하되, 증강은 제외하고 평가에만 사용합니다.

- **DataLoader 설정**
  - 클라이언트: `batch_size=32`, `shuffle=True`로 설정하여 학습용 배치 구성
  - 서버(테스트): `batch_size=64`, `shuffle=False`로 전 데이터에 대해 평가

---

### 4. 학습/추론 최적화 & 재현성

단순히 학습이 돌아가는 수준을 넘어서, **Edge 환경을 고려한 최적화와 재현성**을 함께 챙겼습니다.

- **Mixed Precision Training (AMP) – 클라이언트**
  - CUDA 디바이스에서만 `torch.amp.autocast('cuda', enabled=use_amp)`와  
    `torch.amp.GradScaler('cuda')`를 사용해 **Mixed Precision 학습**을 수행합니다.
  - Forward/Backward를 FP16/FP32 혼합으로 처리함으로써:
    - 연산 속도 향상
    - GPU 메모리 사용량 감소
  - CUDA가 아닌 경우에는 AMP를 비활성화하고 **일반 FP32 학습**으로 안전하게 폴백합니다.

- **FP16 Inference – 서버**
  - 서버에서 글로벌 모델 성능을 측정할 때:
    - `model.half()`로 모델 파라미터를 FP16으로 변환하고,
    - 입력도 `inputs.to(device).half()`로 맞춰 **Half Precision 추론**을 수행합니다.
  - 이를 통해 테스트셋 전체에 대한 **추론 시간을 단축**하고,  
    실험적으로 Edge/Jetson 환경에서의 inference 최적화 전략을 반영합니다.

- **Optimizer, Loss, 정규화**
  - Optimizer: `Adam(lr=0.001, weight_decay=1e-4)`
    - `weight_decay`로 L2 규제를 걸어 과적합을 완화합니다.
  - 손실함수: `CrossEntropyLoss(weight=class_weights)`
    - `class_weights = [3.1, 4.0, 2.5, 3.3]`와 같이 클래스별 다른 가중치를 부여해,
      클래스 불균형 환경에서 **소수 클래스의 영향력을 보정**했습니다.

- **디바이스 선택 & 폴백 전략**
  - 실행 시점에:
    - `torch.backends.mps.is_available()` → `cuda.is_available()` → 그 외에는 `cpu`
  - 순서대로 체크해 디바이스를 자동으로 선택하여,
    - Apple Silicon(MPS),
    - NVIDIA GPU(CUDA),
    - 일반 CPU 환경
    모두에서 코드 수정 없이 동작하도록 했습니다.

- **재현성을 위한 시드 및 설정**
  - `SEED = 42`로 고정하고,
    - `random.seed(SEED)`
    - `np.random.seed(SEED)`
    - `torch.manual_seed(SEED)`
    - `torch.cuda.manual_seed_all(SEED)`
    를 설정했습니다.
  - 또한:
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`
    로 설정해, **속도 최적화보다 결과 일관성을 우선**하도록 구성했습니다.
  - 이를 통해 같은 설정에서 다시 실행했을 때, **가급적 동일한 학습 경향과 성능이 재현**되도록 했습니다.

<br>

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

<br>

## 📊 결과(Jetson)


- 학습 성능 : 80.01525553012968 %

- 학습 소요 시간: 0 시간 0 분 54.24 초

- 최종 모델 크기: 0.1158 MB

- 예측 소요 시간 : 0.70 초
