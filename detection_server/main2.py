from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse, StreamingResponse
from ultralytics import YOLO
from contextlib import asynccontextmanager
import cv2
import os
import subprocess as sp
import time
from PIL import Image
import torch

# 환경 변수 설정
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# YOLO 모델 로드
model1 = YOLO("yolov8n.pt")  # YOLOv8 모델을 로드합니다.

# ffmpeg 시작 함수
def start_ffmpeg():
    sp.Popen("/home/user1/logo-detection/detection_server/mediamtx")

    input_mp4 = '/home/user1/logo-detection/yolov8/video.mp4'
    output_rtsp_url = 'rtsp://127.0.0.1:8554/video0'

    ffmpeg_command = [
        'ffmpeg',
        '-re',
        '-i', input_mp4,
        '-flags', '+bitexact',
        '-qscale:v', '2',
        '-b:v', '2000k', 
        '-rtsp_transport', 'tcp',  # RTSP 전송 프로토콜 설정
        '-max_delay', '0',
        '-f', 'rtsp', output_rtsp_url,
    ]

    sp.Popen(ffmpeg_command, stdin=sp.PIPE)

    return PlainTextResponse("server is running")

# RTSP 또는 비디오 스트림 소스 설정
cap = cv2.VideoCapture(1)  # 사용하려는 카메라 인덱스를 2로 설정 (Logitech 웹캠을 사용하려는 경우)

print(cap.isOpened())  # 카메라 또는 비디오 소스가 열렸는지 확인

img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print("RTSP(or video) open failed")
    exit()

# GPU가 있는지 확인하고 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 메인 루프: 비디오 스트림을 읽고 YOLO 모델 적용
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLO 모델에 프레임을 입력, CPU 또는 GPU에서 실행
    results = model1(
        frame, 
        device=device,  # 자동으로 선택된 디바이스를 사용
        conf=0.77, 
        iou=0.2, 
        augment=True, 
        verbose=False, 
        imgsz=640,
    )

    # 결과 처리: 첫 번째 결과의 바운딩 박스와 라벨 정보
    for result in results:
        boxes = result.boxes  # 바운딩 박스 정보
        
        # 바운딩 박스를 영상에 그리기
        for box in boxes:
            # 바운딩 박스 좌표 (int로 변환)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 클래스 라벨과 confidence 추출
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            label = model1.names[class_id]  # 클래스 라벨 이름 가져오기
            
            # 바운딩 박스 그리기 (빨간색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 라벨과 confidence 텍스트 그리기
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 인식 결과를 화면에 출력 (원하는 경우 제거 가능)
    cv2.imshow('YOLO Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 스트림이 끝나면 캡처 릴리즈 및 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
