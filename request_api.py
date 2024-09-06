import requests

# Define the API endpoint
url = "http://fastapi-env.eba-tnd8hdvx.us-east-1.elasticbeanstalk.com/EV_response"

# Define the query parameters (user_input)
params = {
    "user_input": "Hello what are the best applications?"
}

# Send the POST request to the FastAPI endpoint with the query parameter
response = requests.post(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    response_data = response.json()
    
    # Extract the "text" field from the response
    ai_response_text = response_data.get("response", {}).get("text")
    
    # Print the extracted text
    print("AI Response Text:")
    print(ai_response_text)
else:
    print(f"Error: {response.status_code} - {response.text}")
