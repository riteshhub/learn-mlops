"""
Lambda function creates an endpoint configuration and deploys a model to real-time endpoint. 
Required parameters for deployment are retrieved from the event object
"""

import json
import boto3
from time import strftime, gmtime

def lambda_handler(event, context):
    print(event)
    sm_client = boto3.client("sagemaker")
    
    current_timestamp = strftime('%m-%d-%H-%M', gmtime())
    # Details of the model created in the Pipeline CreateModelStep
    base_name = "loan-classification-xgboost"
    model_name = f"{base_name}-model-{current_timestamp}"
    endpoint_name = f"{base_name}-endpoint-{current_timestamp}"
    endpoint_config_name = f"{base_name}-config-{current_timestamp}"
    model_package_arn = event["detail"]["ModelPackageArn"]
    role = "dummy_role" # add your role arn
    instance_type = event["detail"]["InferenceSpecification"]["SupportedRealtimeInferenceInstanceTypes"][0]
    instance_count = 1
    primary_container = {"ModelPackageName": model_package_arn}
    
    
    print("model name - ",model_name)
    print("endpoint_name - ",endpoint_name)
    print("endpoint_config_name - ",endpoint_config_name)
    print("model_package_arn - ",model_package_arn)
    print("role - ",role)
    print("instance_type - ",instance_type)
    print("primary_container - ",primary_container)

    # Create model
    model = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer=primary_container,
        ExecutionRoleArn=role
    )

    # Create endpoint configuration
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "Alltraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": "s3_path", # add your s3 path where you want to store end users request and response
            "CaptureOptions": [
                    {'CaptureMode': 'Input'}, 
                    {'CaptureMode': 'Output'}
            ]
        }
    )

    # Create endpoint
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, 
        EndpointConfigName=endpoint_config_name
    )