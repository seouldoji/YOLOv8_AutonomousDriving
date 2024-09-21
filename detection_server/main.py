from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse, StreamingResponse
from ultralytics import YOLO
from contextlib import asynccontextmanager
import cv2
import os
import subprocess as sp
from ensemble_boxes import weighted_boxes_fusion, soft_nms, non_maximum_weighted
import time
from PIL import Image

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # rtsp로 받아오기

app = FastAPI()
# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_500_m:2023-11-153/weights/best.pt")

# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_500_l:2023-11-17/weights/best.engine") #logo_backup/

# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_1000_l:2023-11-22/weights/best.pt")
# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_augmentation_640_300_l:2023-11-21_auto/weights/best.engine")
# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_augmentation_640_300_l:2023-11-21_auto_grayscale/weights/best.engine")

# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_100_l:2023-11-21/weights/best.onnx") #100 epochs

# model = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_500_l:2023-11-23/weights/best.pt")  # now best

model1 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_500_l:2023-11-23/weights/best.pt")  # now best
model2 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_640_300_l:2023-11-20_now_best/weights/best.pt")
# model3 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_1280_200_l:2023-11-192/weights/best.engine")
model3 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_1280_500_m:2023-11-14/weights/best.pt") # 학습끝나면 마지막으로 이거 앙상블 돌려보기 
model3 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_1280_500:2023-11-13/weights/best.pt")
model3 = YOLO("/home/user1/logo-detection/yolov8/runs/detect/b_logo_no_augmentation_1920_500_l:2023-11-28/weights/best.pt")


@app.get("/")
def hello():
  return PlainTextResponse("server is running")

@app.get("/rtmp_setting")
def start_ffmpeg():

  sp.Popen("/home/user1/logo-detection/detection_server/mediamtx")

  input_mp4 = '/home/user1/logo-detection/yolov8/video.mp4'
  input_mp4 = '/home/user1/logo-detection/yolov8/video1.mov'
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

def get_stream():
  # cap = cv2.VideoCapture(")  # 비디오 스트림 소스를 지정합니다.
  url = "/home/user1/logo-detection/yolov8/video.mp4"
  url = "rtsp://127.0.0.1:8554/video0"
  url = "/home/user1/logo-detection/yolov8/video1.mov"

  # cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
  cap = cv2.VideoCapture(url)
  img_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  img_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  if not cap.isOpened():
    print("RTSP(or video) open failed")

  frame_count = 0
  # start_time = time.time()
  # fps = 0

  

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # img norm
    # frame = cv2.add(frame, (-20, -20, -20, 0))

    frame_count += 1
 
    # YOLOv8 모델을 이용해 객체 탐지를 수행합니다.  
    
    # one model predict
    # frame = Image.fromarray(frame).convert("L") # grayscale
    # results = model.track(
    #   frame, 
    #   device=[0, 1], 
    #   conf=0.25, 
    #   iou=0.5, 
    #   # augment=True, 
    #   verbose=False, 
    #   # half=True, 
    #   tracker='./bot_sort.yaml',
    #   imgsz=640,
    #   persist=True,
    #   )

    # annotated_frame = results[0].plot()
    # frame = annotated_frame
    

    # model ensemble
    result1 = model1(
      frame, 
      device=[0, 1], 
      conf=0.77, 
      iou=0.2, 
      augment=True, 
      verbose=False, 
      # half=True, 
      imgsz=640,
      )
    
    result2 = model2(
      frame, 
      device=[0, 1], 
      conf=0.25, 
      iou=0.2, 
      # augment=True, 
      verbose=False, 
      # half=True, 
      imgsz=640,
      )
    
    # result3 = model1(
    #   frame, 
    #   device=[0, 1], 
    #   conf=0.77, 
    #   iou=0.5, 
    #   augment=False, 
    #   verbose=False, 
    #   # half=True, 
    #   imgsz=640,
    #   )
    
    result3 = model3(
      frame, 
      device=[0, 1], 
      conf=0.25, 
      iou=0.5, 
      # augment=True, 
      verbose=False, 
      # half=True, 
      imgsz=1920,
      )


    result1 = result1[0].boxes
    result2 = result2[0].boxes
    result3 = result3[0].boxes
    # result4 = result4[0].boxes


    box_list = [
      result1.xyxyn.tolist(), result2.xyxyn.tolist(), result3.xyxyn.tolist() #, result4.xyxyn.tolist()
    ]

    score_list = [
      result1.conf.tolist(), result2.conf.tolist() , result3.conf.tolist() #, result4.conf.tolist()
    ]

    label_list = [
      list(map(int, result1.cls.tolist())), list(map(int, result2.cls.tolist())) , list(map(int, result3.cls.tolist())) #, list(map(int, result4.cls.tolist()))
    ] 

    # print(box_list)
    # print(score_list)
    # print(label_list)

    weights = [ 3, 2, 1]
    iou_thr = 0.5
    skip_box_thr = 0.25

    boxes, confs, classes = weighted_boxes_fusion(box_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # print(boxes, confs, cls)

    col_red = (0, 0, 255)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
      x1, x2, y1, y2, conf = x1 * img_w, x2 * img_w, y1 * img_h, y2 * img_h, confs[idx]
      x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
      # print([x1, x2, y1, y2], conf)
      frame = cv2.putText(frame, f"{conf:.6f}", (x1, y1 - 20), cv2.FONT_ITALIC, 1, col_red, 2) if conf > 0.25 else frame
      frame = cv2.rectangle(frame, (x1, y1), (x2, y2), col_red, 2)  if conf > 0.25 else frame

    boxes, confs, classes = non_maximum_weighted(box_list, score_list, label_list, weights=weights, iou_thr=iou_thr)
    # print(boxes, confs, cls)

    col_green = (0, 255, 0)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
      x1, x2, y1, y2, conf = x1 * img_w, x2 * img_w, y1 * img_h, y2 * img_h, confs[idx]
      x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
      # print([x1, x2, y1, y2], conf)
      frame = cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 20), cv2.FONT_ITALIC, 1, col_green, 2) if conf >= 0.25 else frame
      frame = cv2.rectangle(frame, (x1, y1), (x2, y2), col_green, 2)  if conf > 0.25 else frame


    # blur
    # for x1, y1, x2, y2 in results[0].boxes.xyxy.tolist():
    #   x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #   roi = frame[y1:y2, x1:x2]   # 모자이크 영역 지정
    #   roi = cv2.blur(roi, (30, 30)) # 블러(모자이크) 처리
    #   frame[y1:y2, x1:x2] = roi

    # fps calculate
    # if frame_count % 30 == 0:
    #   end_time = time.time()
    #   elapsed_time = end_time - start_time
    #   fps = frame_count / elapsed_time

    #   start_time = time.time()
    #   frame_count = 0
    # cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 이미지를 인코딩하여 웹소켓을 통해 전송합니다.
    _, buffer = cv2.imencode('.jpg', frame)
    jpgBin = bytearray(buffer.tobytes())
    yield (b'--PNPframe\r\n' b'Content-type: image/jpeg\r\n\r\n' + jpgBin)

    
@app.get("/video_feed")
def video():
  return StreamingResponse(get_stream(), media_type="multipart/x-mixed-replace; boundary=PNPframe")

