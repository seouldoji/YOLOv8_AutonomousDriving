from ultralytics import YOLO
import yaml
from time import time, strftime

def create_yaml(logo_or_sign):
  data = {
     "path": '/home/user1/logo-detection/dataset/'+logo_or_sign,
      "train" : 'train',
      "val" : 'valid',
      "names" : {0 : 'brand_logo'}
    }

  with open(f'./brand_logo.yaml', 'w') as f :
      yaml.dump(data, f)  

  # check written file
  with open(f'./brand_logo.yaml', 'r') as f :
      lines = yaml.safe_load(f)
      print(lines)

if __name__ == "__main__":
   # create yaml
  logo_or_sign = "logo"
  create_yaml(logo_or_sign)

  epochs = 500
  imgsz = 1920
  augmentation = False
  size = 'l'
  optimizer = 'AdamW'
  lr = 0.001
  batch = 8
  model = f'yolov8{size}.pt'
  model = YOLO(model)
  ext = "engine"


  aug = "augmentation" if augmentation == True else "no_augmentation"


  if augmentation == False:
    result = model.train(
    data='brand_logo.yaml', 
    epochs=epochs, 
    device=[0, 1], 
    optimizer=optimizer,
    name=f"b_logo_{aug}_{imgsz}_{epochs}{''.join(['_',size])}:{strftime('%Y-%m-%d')}",
    imgsz=imgsz,
    lr0=lr,
    cos_lr=True,
    batch=batch,
    format=ext
    )
  else:
    result = model.train(
    data='brand_logo.yaml', 
    epochs=epochs, 
    device=[0, 1], 
    optimizer=optimizer,
    name=f"b_logo_{aug}_{imgsz}_{epochs}{''.join(['_',size])}:{strftime('%Y-%m-%d')}_{optimizer}_all_aug",
    imgsz=imgsz,
    lr0=lr,
    cos_lr=True,
    batch=batch,
    format=ext,
    # resume=True,

    hsv_h= 0.015,  # image HSV-Hue augmentation (fraction) 이미지 색조
    hsv_s= 0.7,  # image HSV-Saturation augmentation (fraction) 이미지 채도
    hsv_v= 0.4,  # image HSV-Value augmentation (fraction) 이미지 명도
    degrees= 0.5,  # image rotation (+/- deg) 이미지 회전
    translate= 0.1,  # image translation (+/- fraction) 이미지 이동
    scale= 0.3,  # image scale (+/- gain) 이미지 크기
    fliplr= 0.5, # image flip left-right (probability) 이미지 좌우반전 확률
    mosaic= 0.3,  # image mosaic (probability) 4개의 이미지 하나로 묶을 확률
    mixup= 0.1  # image mixup (probability) 두 이미지 선형적으로 섞을 확률
    )
    
  model.export(format=ext)
