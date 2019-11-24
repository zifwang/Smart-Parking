import requests
import base64
import json

image_path = 'MICHIGAN.jpg'
SECRET_KEY = 'sk_097b1a9efe75fbd0ca0d9cd1'

with open(image_path,'rb') as image_file:
    image_base64 = base64.b64encode(image_file.read())

url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
r = requests.post(url, data = image_base64)

car_info = r.json()

license_plate = car_info['results'][0]['plate']
region = car_info['results'][0]['region']

print('License_plate: ' + license_plate)
print('State: ' + region)