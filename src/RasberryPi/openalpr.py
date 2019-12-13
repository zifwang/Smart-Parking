import requests
import base64
import json
import os
import time
import sys
import paho.mqtt.client as mqtt


image_path = 'MICHIGAN.jpg'
SECRET_KEY = 'sk_097b1a9efe75fbd0ca0d9cd1'

with open(image_path,'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read())

url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
r = requests.post(url, data = image_base64)

car_info = r.json()
print(car_info)

license_plate = car_info['results'][0]['plate']
region = car_info['results'][0]['region']

print('License_plate: ' + license_plate)
print('State: ' + region)

# # Data Capture and upload interval in seconds.
# INTERVAL = 2

# sensor_data = {'license_plate':'','time':''}

# client = mqtt.Client()

# ACCESS_TOKEN = 'PHbFRbbdKLzHP0JVLgBd'
# THINGSBOARD_HOST = '54.193.127.127'

# # Set access token
# client.username_pw_set(ACCESS_TOKEN)

# # Connect to ThingsBoard using default MQTT port and 60 seconds keepalive interval
# client.connect(THINGSBOARD_HOST, 1883, 60)

# client.loop_start()

# sensor_data['license_plate'] = license_plate
# # Sending humidity and temperature data to ThingsBoard
# client.publish('v1/devices/me/telemetry', json.dumps(sensor_data), 1)

# client.loop_stop()
# client.disconnect()