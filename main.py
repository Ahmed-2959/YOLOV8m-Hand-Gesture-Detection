import torch
import numpy as np
from ultralytics import YOLO
import cv2
import os
import time

model = YOLO("./last.pt")
cam = cv2.VideoCapture(0)

last_executed = 0        
Audio_Pause_Buffer = 1.5  

while True:
    ret, frame = cam.read()
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
    results = model.predict(frame)
    names = model.names

    for result in results:
            for r in result.boxes :
                x1, y1, x2, y2 = r.xyxy[0]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                score = float(r.conf[0])
                arr = np.array([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (x1, y1), (x2,y2), 3)
            for c in result.boxes.cls:
                cv2.putText(frame, f"Class:{names[int(c)]}", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)

                if names[int(c)] == "Thumbs up":
                    os.system("xdotool key XF86AudioRaiseVolume")

                elif names[int(c)] == "Thumbs Down":
                    os.system("xdotool key XF86AudioLowerVolume")

                elif names[int(c)] == "Stop": 
                    current_time = time.time()
                    if current_time - last_executed >= Audio_Pause_Buffer:
                        os.system("xdotool key XF86AudioPlay")
                        last_executed = current_time

        

    cv2.imshow('image', frame)
    cv2.waitKey(25)
cam.release()
cv2.destroyAllWindows()
