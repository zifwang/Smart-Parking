import json
import boto3    
print('Loading function')


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    # line 9 - 15 store data to s3
    if 'license_plate' in event:
        print("got license_plate")
        license = boto3.resource('s3') # license can be replaced  
        s3object = license.Object('parkingdataaa', 'license.json') # s3object can be replaced

        s3object.put(
            Body=(bytes(json.dumps(event).encode('UTF-8')))
        )
    
    if 'id' in event:
        print("got xdot")
        xdot = boto3.resource('s3') # license can be replaced  
        
        if '0' in event['id']:
            s3object = xdot.Object('parkingdataaa', 'xdot1.json') # s3object can be replaced

            s3object.put(
                Body=(bytes(json.dumps(event).encode('UTF-8')))
            )
        
        if '1' in event['id']:
            s3object = xdot.Object('parkingdataaa', 'xdot2.json') # s3object can be replaced

            s3object.put(
                Body=(bytes(json.dumps(event).encode('UTF-8')))
            )
        
        if '2' in event['id']:
            s3object = xdot.Object('parkingdataaa', 'xdot3.json') # s3object can be replaced

            s3object.put(
                Body=(bytes(json.dumps(event).encode('UTF-8')))
            )
        
        
    #raise Exception('Something went wrong')
