🚀 Jetson Federated Learning for Medical MRI Classification

경량 CNN 모델을 이용한 Jetson 기반 연합학습(Federated Learning) 프로젝트입니다.
두 개의 디바이스(Jetson Orin Nano)가 서로 다른 MRI 데이터를 공유하지 않고,
서버를 통해 FedAvg 방식으로 모델을 지속적으로 개선하는 시스템을 구축했습니다.

📌 프로젝트 개요

본 프로젝트는 여러 병원이 보유한 환자의 MRI 데이터를 서버로 직접 전송하지 않고,
개별 디바이스에서 학습한 모델만 공유하는 프라이버시 보장 인공지능 시스템을 목표로 합니다.

🎯 목적

Jetson Orin Nano 2대를 활용한 분산 학습 환경 구축

Non-IID MRI 데이터 기반 4-클래스 분류

FedAvg 알고리즘을 이용한 글로벌 모델 업데이트

경량 모델 설계 및 성능 최적화

🧩 시스템 구성
🔹 Server (server.py)

클라이언트 모델 파라미터 수신

FedAvg 수행

글로벌 모델 재배포

🔹 Client 1 (client1.py)

Glioma·Meningioma 중심 Non-IID 데이터 포함

로컬 학습 후 서버로 weight 전송

🔹 Client 2 (client2.py)

Notumor·Pituitary 중심 Non-IID 데이터 포함

로컬 학습 후 서버와 반복적 통신

🧠 데이터셋 구성

제공된 .pt 파일 기반 MRI 4-class 분류 데이터:

Label	Class
0	Glioma
1	Meningioma
2	Notumor
3	Pituitary

각 클라이언트는 서로 다른 비율의 데이터(non-IID)를 보유하며,
이는 연합학습 시 정확도 및 안정성에 영향을 줍니다.

📁 프로젝트 구조
jetson-fedlearning
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

CNN 기반 초경량 모델

입력: 3 × 192 × 192

Conv → BatchNorm → ReLU 블록 반복

AdaptiveAvgPool2d + Linear

Jetson에서도 학습·추론 가능한 경량 구조

🔄 Federated Learning (FedAvg) 흐름
Client1 ----→
              \
               →—— Server ——→ Global Update → Clients
              /
Client2 ----→

수행 절차

각 Client에서 로컬 학습 수행

학습된 weight를 Server로 전송

Server에서 FedAvg로 글로벌 weight 계산

갱신된 모델을 클라이언트에게 재배포

목표 정확도 또는 설정된 라운드 종료 시 학습 완료

📊 평가 지표
항목	설명
Accuracy	Test set 기반 정확도
Training Time	Jetson에서의 학습 소요 시간
Model Size	최종 모델 파라미터 크기(MB)
Inference Time	Jetson에서의 1회 추론 시간
🔧 실행 방법
1) 서버 실행
python server.py

2) Client 1 실행
python client1.py

3) Client 2 실행
python client2.py


서버와 클라이언트는 Socket 기반 통신을 사용하며,
각 라운드마다 글로벌 모델이 자동으로 갱신됩니다.

🧪 결과 예시

(실제 실험 결과에 맞게 수정하세요)

Global round [4/5] Accuracy: 79.7%

Training Time: 1 min 40 sec

Model Size: 0.112 MB

Inference Time: 2.24 sec

🏁 마무리

본 프로젝트는 Jetson과 연합학습을 결합하여
프라이버시 보호 + 분산 학습 + 경량 모델 최적화라는 3가지 목표를 성공적으로 달성하는 것을 목표로 합니다.
