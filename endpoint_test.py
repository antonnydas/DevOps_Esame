runtime_client = boto3.client('sagemaker-runtime', region_name=aws_region)

payload = {
    "Pclass": 0.5,
    "Sex": 0.3
}

response = runtime_client.invoke_endpoint(
    EndpointName='prismaelettrolitico',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(result)
