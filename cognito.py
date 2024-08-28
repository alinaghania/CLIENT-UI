import boto3

def authenticate_user(username, password):
    client = boto3.client('cognito-idp')
    
    response = client.initiate_auth(
        ClientId='your_cognito_client_id',
        AuthFlow='USER_PASSWORD_AUTH',
        AuthParameters={
            'USERNAME': username,
            'PASSWORD': password
        }
    )
    
    return response['AuthenticationResult']['AccessToken']

# Example usage:
access_token = authenticate_user('username', 'password')
