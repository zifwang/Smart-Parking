import json
import boto3
def lambda_handler(event, context):
    # TODO implement
    # Read license plate
    plate_s3 = boto3.resource('s3')
    plate_object = plate_s3.Object('parkingdataaa', 'license.json')
    file_content = plate_object.get()['Body'].read().decode('utf-8')
    plate_content = json.loads(file_content)
    plate = plate_content['license_plate']
    # Read xdot1
    xdot1_s3 = boto3.resource('s3')
    xdot1_object = xdot1_s3.Object('parkingdataaa', 'xdot1.json')
    file_content = xdot1_object.get()['Body'].read().decode('utf-8')
    xdot1_content = json.loads(file_content)
    
    # Read xdot2
    xdot2_s3 = boto3.resource('s3')
    xdot2_object = xdot2_s3.Object('parkingdataaa', 'xdot2.json')
    file_content = xdot2_object.get()['Body'].read().decode('utf-8')
    xdot2_content = json.loads(file_content)
    
    # Read xdot3
    #xdot3_s3 = boto3.resource('s3')
    #xdot3_object = xdot3_s3.Object('parkingdataaa', 'xdot3.json')
    #file_content = xdot3_object.get()['Body'].read().decode('utf-8')
    #xdot3_content = json.loads(file_content)
    
    #combine signals
    combine_xdot = [xdot1_content['location'], xdot2_content['location']]
    pattern = "".join(combine_xdot)
    decision = location_decision(pattern)
    
    print("License Plate and Parking Position:")
    print(plate + ": position " + decision)
    
    pass

def location_decision(input):

    position = 0
    if input == "00":
        position = "Floor 1"
    elif input == "10":
        position = "Floor 2"
    elif input == "01":
        position = "Floor 3"
    elif input == "11":
        position = "Floor 3"

    return position

    pass



    
   