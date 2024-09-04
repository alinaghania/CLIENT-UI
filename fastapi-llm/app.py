from fastapi import FastAPI, HTTPException
from utils import initialize_chain, add_message_to_history
from botocore.exceptions import ClientError
import os

# Initialize FastAPI
app = FastAPI()

# Check the current working directory
print(f"Current working directory: {os.getcwd()}")  # <-- Print the current working directory

# Route to get a response from the Claude model
@app.post("/EV_response")
async def get_response(user_input: str):
    try:
        # Initialize the chain with user input
        chain = initialize_chain()

        # Invoke the chain to process the input and get the response
        response = chain.invoke({"input": user_input})

        # Add the user input and AI response to the chat history
        add_message_to_history("human", user_input)
        add_message_to_history("assistant", response)

        return {"response": response}

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"AWS Client Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
