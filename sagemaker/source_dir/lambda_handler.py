import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    #data = json.loads(json.dumps(event))
    #payload = data['data']
    #payload=event["text"]
    #print(payload)
    # result = []
    # for input in payload:
    #     serialized_input = ','.join(map(str,input))
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=json.dumps(event))
    #print(response)
    result=response['Body'].read().decode()
    print(result)
    #pred = result
    predicted_label = 'circRNA' if result == 1 else 'lincRNA'
    
    return predicted_label