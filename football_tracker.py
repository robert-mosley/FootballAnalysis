from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import pyttsx3
import threading
import math
import sys

voice = pyttsx3.init()
model = YOLO("best.pt")
white_player = 0
green_player = 0
locations = []
player_distance = {}
player_coordinates = ()

i = 0

direction = "unknown"

def get_center(x1,x2,y1,y2):
    x = (x1+x2) / 2
    y = (y1+y2) / 2
    return x, y

def get_distance(x1,x2,y1,y2):
    x = x2 - x1
    y = y2 - y1
    return math.sqrt((x**2 + y**2))

def add(x):
    global locations
    if len(locations) == 4:
        locations.pop(0)
    locations.append(x)

def speak(words):
    voice.say(words)
    voice.runAndWait()

def speak_thread(words):
    threading.Thread(target=speak, args=(words,), daemon=True).start()

file = r"example.mp4"

video = cv2.VideoCapture(file)
start = time.time()
start2 = time.time()
foundAt = 0
shortest = 200

while True:
    green_player = 0
    white_player = 0
    players = []
    ball_coordinates = None
    ret, img = video.read()
    results = model.predict(img)

    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        ball_coordinates = (x1,y1,x2,y2)
        if int(box.cls[0]) == 1:
            ballx, bally = get_center(x1,x2,y1,y2)
            add(ballx)

            if len(locations) >= 2:
                dx = np.mean(np.diff(locations[-3:]))
                if dx < 0:
                    direction = "Right"
                else:
                    direction = "Left"
        
        elif int(box.cls[0]) == 2:
            player_coordinates = player_coordinates + ((x1,x2,y1,y2),)
            cx, cy = get_center(x1, x2, y1, y2)

            player_crop = img[y1:y2, x1:x2]
            hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
            hsv_matrix = hsv_crop.reshape(-1,3)
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(hsv_matrix)
            centers = kmeans.cluster_centers_
            labels, counts = np.unique(kmeans.labels_, return_counts=True)

            dominant = centers[np.argmax(counts)]
            h, s, v = dominant

            if 35 < h < 85 and s > 40:
                h, s, v = centers[np.argsort(counts)[-2]]

            if s < 40 and v > 200:
                white_player += 1
                players.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "label": "white"
                })
                """
                plt.imshow(player_crop)
                plt.show()
                """
            elif 35 < h < 85:
                green_player += 1
                players.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "label": "green"
                })
                """
                plt.imshow(player_crop)
                plt.show()
                """
    shortest = float("inf")
    nearest = None

    for p in players:
        px, py = p["center"]
        d = get_distance(ballx, px, bally, py)
        if d < shortest:
            shortest = d
            nearest = p

    if nearest:
        x1, y1, x2, y2 = nearest["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"Player ({nearest['label']})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if ball_coordinates:
        x1, y1, x2, y2 = ball_coordinates[0], ball_coordinates[1], ball_coordinates[2], ball_coordinates[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, "Ball", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    print(green_player, white_player)
    elapsed = time.time() - start
    elapsed_time = time.time() - start2
    cv2.putText(img, f"Ball direction: {direction}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    cv2.putText(img, f"White Jersey Players {white_player}", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    cv2.putText(img, f"Green Jersey Players: {green_player}", (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    cv2.putText(img, f"Time elapsed: {round(elapsed_time,2)}", (50,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    
    if elapsed > 60: 
        if direction == "Left":
            speak_thread("White is currently attacking")
            start = time.time()
        elif direction == "Right":
            speak_thread("Green is currently attacking")
            start = time.time()

    cv2.imshow("Game",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

