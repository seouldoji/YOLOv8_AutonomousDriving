from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse, StreamingResponse
from ultralytics import YOLO
from contextlib import asynccontextmanager
import cv2
import os
import subprocess as sp
import time
from PIL import Image

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # rtsp로 받아오기

model1 = YOLO("yolov8n.pt")  # now best


def start_ffmpeg():

  sp.Popen("/home/user1/logo-detection/detection_server/mediamtx")

  input_mp4 = '/home/user1/logo-detection/yolov8/video.mp4'
  output_rtsp_url = 'rtsp://127.0.0.1:8554/video0'

  ffmpeg_command = [
      'ffmpeg',
      # '-stream_loop', '-1',
      # '-loglevel', 'panic',
      '-re',
      '-i', input_mp4,
      '-flags', '+bitexact',
      # '-r', '20',
      '-qscale:v', '2',
      '-b:v', '2000k', 
      # '-s', '1280x720', 
      '-rtsp_transport', 'tcp',  # RTSP 전송 프로토콜 설정
      '-max_delay', '0',
      '-f', 'rtsp', output_rtsp_url,
  ]

  sp.Popen(ffmpeg_command, stdin=sp.PIPE)

  return PlainTextResponse("server is running")

# start_ffmpeg()


# cap = cv2.VideoCapture(")  # 비디오 스트림 소스를 지정합니다.
url = "/home/user1/logo-detection/yolov8/video.mp4"

# cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0)

print(cap.isOpened())

# exit()

img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
  print("RTSP(or video) open failed")

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # model ensemble
  result = model1(
    frame, 
    device=[0, 1], 
    conf=0.77, 
    iou=0.2, 
    augment=True, 
    verbose=False, 
    # half=True, 
    imgsz=640,
    )
  


  result = result[0].boxes

  # 아두이노에서 초음파 + 물체가 얼마나 

  # 차선 상황 + 앞쪽 장애물 상황
  
  # 차 어떻게 움직일 건지