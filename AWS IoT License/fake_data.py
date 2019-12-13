from __future__ import print_function
import numpy as np
import cv2
import requests
import base64
import json
import os
import time
import sys
import paho.mqtt.client as mqtt
import ssl
import time
import datetime
import logging, traceback
from datetime import datetime

SECRET_KEY = 'sk_097b1a9efe75fbd0ca0d9cd1'
def get_license_plate_openalpr(image):
	with open(image,'rb') as image_file:
		image_base64 = base64.b64encode(image_file.read())
	url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
	r = requests.post(url, data = image_base64)
	car_info = r.json()

	if 'results' in car_info:
		if car_info['results']:
			return True, car_info['results'][0]['plate']

	return False, ""


if __name__ == '__main__':
    month_dict = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}

    mqttc = mqtt.Client()
    ACCESS_TOKEN = 'PHbFRbbdKLzHP0JVLgBd'
    THINGSBOARD_HOST = '54.193.127.127'
    mqttc.username_pw_set(ACCESS_TOKEN)
    mqttc.connect(THINGSBOARD_HOST,1883,60)
    mqttc.loop_start()
    count = 0
    sensor_data = {}

    # Set Camera
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
    	ret, frame = cap.read()
    else:
    	ret = False
        
    while ret:
        # Capture Frame
        ret, frame = cap.read()
        cv2.imwrite('car.jpg',frame)
        # Displaythe resulting frame
        flag, license_plate = get_license_plate_openalpr('car.jpg')
        if flag:
            now = datetime.now().strftime('%m %d %Y %H:%M:%S')
            month = month_dict[now[0:2]]
            now = month+now[2:]
            location = str(count)
            sensor_data['location'] = location
            sensor_data['license_plate'] = license_plate
            sensor_data['datetime'] = now
            mqttc.publish('v1/devices/me/telemetry', json.dumps(sensor_data),1)
            count = count + 1
            time.sleep(10)

        cv2.imshow("preview",frame)
        if cv2.waitKey(1) and 0xFF==ord('q'):
            break
    
    cv2.destroyWindow("preview")
    cap.release()
    mqttc.loop_stop()
    mqttc.disconnect()


