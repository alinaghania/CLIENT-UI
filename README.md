
# EV Genius - Expert en véhicules électriques Peugeot



## SET UP 

1. **Install AWS CLI**:

    On macOS, run the following command in the terminal:

    ```
    brew install awscli
    ```

    On Windows, go to the AWS CLI website and follow the instructions to install it.

2. **Configure AWS CLI**

    ```
    aws configure
    ```

    **You'll be prompted to enter**:
    - AWS Access Key ID [None]: xxx cf antonio
    - AWS Secret Access Key [None]: xxx cf antonio
    - Default region name [None]: us-east-1
    - Default output format [None]: json

3. **Install a model**

    ```
    pip install anthropic==0.28.1
    ```

4. **This command will list the available foundation models under the anthropic provider in the specified region.**

    ```
    aws bedrock list-foundation-models --region=us-west-2 --by-provider anthropic --query "modelSummaries[*].modelId"
    ```

    The output should look like:

    ```
    [
    "anthropic.claude-instant-v1:2:100k",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2:0:18k",
    "anthropic.claude-v2:0:100k",
    "anthropic.claude-v2:1:18k",
    "anthropic.claude-v2:1:200k",
    "anthropic.claude-v2:1",
    "anthropic.claude-v2",
    "anthropic.claude-3-sonnet-20240229-v1:0:28k",
    "anthropic.claude-3-sonnet-20240229-v1:0:200k",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0:48k",
    "anthropic.claude-3-haiku-20240307-v1:0:200k",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0:12k",
    "anthropic.claude-3-opus-20240229-v1:0:28k",
    "anthropic.claude-3-opus-20240229-v1:0:200k",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0:18k",
    "anthropic.claude-3-5-sonnet-20240620-v1:0:51k",
    "anthropic.claude-3-5-sonnet-20240620-v1:0:200k",
    "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ]
    ```

5. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```


## 2. TEST the UI -Streamlit  

6. Run the v0 - chatbot with metrics (token, time)

    ```
    cd stella
    streamlit run 2_chatbot.py
    ```


## 3.TEST THE API

1. Structure
    ```
    fastapi-llm/

    ├── app.py  # Point d'entrée de l'API - Endpoints
    ├── utils.py # Fonctions utilitaires pour l'api

    ```

2. Execution api

    ```
    cd fatsapi-llm
    uvicorn app:app -- reload

    ```

## Build docker image

1.RUN CONTAINER AND IMAGE

    ```
    cd stella
    docker build -t fastapi-llm .
    docker run -v ~/.aws:/root/.aws:ro -p 8000:8000 fastapi-llm
    ```
    Acceder à http://localhost:8000/docs si ca run all good!!
    

