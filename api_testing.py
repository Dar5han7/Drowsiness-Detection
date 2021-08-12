import cv2
import requests
import json
import socket
import sys

count = -1
HOST = '127.0.0.1'
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    # print(image)
    count = count +1
    print (count,"count")
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    cv2.imwrite("frame.jpg" ,image)

    url = "http://127.0.0.1:5000"
    files = {'file': open("frame.jpg", 'rb')}


    requests.post(url, files=files)