# Jetson Federated Learning for Medical MRI Classification

경량 CNN 모델을 이용한 Jetson 기반 연합학습(Federated Learning) 프로젝트입니다.
두 개의 디바이스(Jetson Orin Nano)가 서로 다른 MRI 데이터를 공유하지 않고,
서버를 통해 FedAvg 방식으로 모델을 지속적으로 개선하는 시스템을 구축했습니다.

🚀 프로젝트 개요

본 프로젝트는 여러 병원이 보유한 환자의 MRI 데이터를 서버로 직접 전송하지 않고,
개별 디바이스에서 학습한 모델만 주고받는 프라이버시 보장 인공지능 시스템을 목표로 합니다.

🎯 목적

Jetson Orin Nano 2대를 활용한 분산 학습 환경 구축

Non-IID MRI 데이터 기반 4-클래스 분류

FedAvg 알고리즘을 이용한 글로벌 모델 업데이트

경량 모델 설계 및 성능 최적화

🧩 시스템 구성

📌 시스템은 다음 3개 요소로 구성됩니다:

Server (server.py)

클라이언트 모델 파라미터 수신

FedAvg 수행

Global model 재배포

Client 1 (client1.py)

Glioma·Meningioma 중심의 Non-IID 데이터 포함

로컬 모델 학습 후 서버에 가중치 전송

Client 2 (client2.py)

Notumor·Pituitary 중심 Non-IID 데이터 포함

로컬 학습 및 서버-클라이언트 라운드 반복 수행

📌 제공된 Jetson 시스템 구성도는 PDF page 5에 상세하게 설명되어 있습니다.

🧠 데이터셋 구성

제공된 .pt 파일 기반 MRI 4-class 분류 데이터

Label 0: Glioma

Label 1: Meningioma

Label 2: Notumor

Label 3: Pituitary

각 클라이언트는 서로 다른 비율의 라벨 데이터(non-IID)를 보유하며,
이는 연합학습 과정에서 성능 차이와 안정성에 영향을 줍니다.

📁 jetson-fedlearning
│
├── server.py
├── client1.py
├── client2.py
│
├── dataset/
│   ├── client1.pt
│   ├── client2.pt
│   └── test.pt
│
├── README.md
└── requirements.txt

⚙️ 모델 구조 (Network1 예시)

CNN 기반 경량 모델

입력: 3×192×192

Conv → BatchNorm → ReLU 반복

AdaptiveAvgPool + Linear Layer

Jetson에서도 학습·추론 가능한 초경량 아키텍처

🔄 연합학습(FedAvg) 흐름
Client1 ----→
              \
               →—— Server ——→ Global Update → Clients
              /
Client2 ----→

Clients는 로컬에서 모델 학습 수행

학습된 weight를 서버에 전송

Server는 FedAvg 수행 후 글로벌 weight 계산

글로벌 모델을 다시 각 클라이언트에게 전송

지정된 round 수 또는 목표 정확도 도달 시 학습 종료
평가 지표 (필수)

항목	설명
학습 성능 (Accuracy)	Test set 기반 정확도
학습 소요 시간 (Training time)	Jetson에서의 실제 학습 시간
모델 크기 (Model Size)	최종 모델의 파라미터 크기 (MB)
추론 시간 (Inference Time)	Jetson에서의 1회 추론 시간

🔧 실행 방법
1. 서버 실행
python server.py

2. Client 1 실행
python client1.py

3. Client 2 실행
python client2.py

서버와 클라이언트는 Socket 기반으로 통신하며,
각 라운드마다 글로벌 모델이 자동으로 교환됩니다.

🧪 결과 예시

(예시 — 실제 실험 결과로 수정 가능)

Global round [4/5] Accuracy: 79.7%

Training Time: 1 min 40 sec

Model Size: 0.112 MB

Inference Time: 2.24 sec
