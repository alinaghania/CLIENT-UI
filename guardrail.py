def guardrail_response(response):
    # Simple guardrail example
    if "inappropriate content" in response.lower():
        return "The response contains inappropriate content and cannot be displayed."
    return response

# Final response to the user
def lambda_handler(event, context):
    user_query = event['queryStringParameters']['query']
    response = process_query(user_query)
    safe_response = guardrail_response(response)
    
    return {
        'statusCode': 200,
        'body': json.dumps({"response": safe_response})
    }
