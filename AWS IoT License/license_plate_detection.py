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
# Import SDK packages
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient




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

IoT_protocol_name = "x-amzn-mqtt-ca"
aws_iot_endpoint = "avtvb2saryiho-ats.iot.us-west-1.amazonaws.com" # <random>.iot.<region>.amazonaws.com
url = "https://{}".format(aws_iot_endpoint)

ca = "AmazonRootCA1.pem" 
cert = "1dd90f1318-certificate.pem.crt"
private = "1dd90f1318-private.pem.key"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(log_format)
logger.addHandler(handler)

def ssl_alpn():
    try:
        #debug print opnessl version
        logger.info("open ssl version:{}".format(ssl.OPENSSL_VERSION))
        ssl_context = ssl.create_default_context()
        ssl_context.set_alpn_protocols([IoT_protocol_name])
        ssl_context.load_verify_locations(cafile=ca)
        ssl_context.load_cert_chain(certfile=cert, keyfile=private)

        return  ssl_context
    except Exception as e:
        print("exception ssl_alpn()")
        raise e

if __name__ == '__main__':
    topic = "thing/camera"
    month_dict = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}

    try:
        mqttc = mqtt.Client()
        ssl_context= ssl_alpn()
        mqttc.tls_set_context(context=ssl_context)
        logger.info("start connect")
        mqttc.connect(aws_iot_endpoint, port=443)
        logger.info("connect success")
        mqttc.loop_start()

        sensor_data = {'license_plate':'','datetime':''}
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
        		sensor_data['license_plate'] = license_plate
        		sensor_data['datetime'] = now
        		logger.info("try to publish:{}".format(json.dumps(sensor_data)))
        		mqttc.publish(topic, json.dumps(sensor_data))
        		time.sleep(10)

        	cv2.imshow("preview",frame)
        	if cv2.waitKey(1) and 0xFF==ord('q'):
        		break

        cv2.destroyWindow("preview")
        cap.release()

    except Exception as e:
        logger.error("exception main()")
        logger.error("e obj:{}".format(vars(e)))
        logger.error("message:{}".format(e.message))
        traceback.print_exc(file=sys.stdout)

