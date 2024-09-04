import contextvars
from pathlib import Path
import boto3
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from botocore.exceptions import ClientError

# Set up a context variable to manage chat history
chat_history_var = contextvars.ContextVar("chat_history", default=[])

# Function to choose the Claude model from Bedrock
def choose_model():
    return ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

# Function to manage memory for conversation
def get_memory():
    return ConversationBufferMemory(return_messages=True)

# Function to initialize the chain with system prompt, context, and memory
def initialize_chain():
    """
    Initialize the conversation chain with system prompt, context, message history, and user input.
    """
    current_directory = Path(__file__).resolve().parent.parent  # Adjust based on your directory structure
    print(f"Current directory for utils.py: {current_directory}")  # <-- Print the current directory

    system_prompt_path = current_directory / "prompt/system_prompt.txt"
    context_path = current_directory / "parsed_data/peugeot_data.txt"

    print(f"System prompt path: {system_prompt_path}")  # <-- Print the path to system_prompt.txt
    print(f"Context path: {context_path}")  # <-- Print the path to peugeot_data.txt

    if not system_prompt_path.exists():
        raise FileNotFoundError("System prompt file not found.")
    if not context_path.exists():
        raise FileNotFoundError("Context file not found.")

    # Read system prompt and context from files
    system_prompt = system_prompt_path.read_text()
    context = context_path.read_text()

    # Replace placeholder {context} in system prompt with actual context content
    formatted_system_prompt = system_prompt.replace("{context}", context)

    # Define the prompt with system prompt and user input
    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        ("human", "{input}")
    ])

    # Get the Bedrock model
    bedrock_llm = choose_model()

    # Create a chain with memory and a prompt
    memory = get_memory()
    chain = LLMChain(
        llm=bedrock_llm,
        prompt=prompt,
        memory=memory,
        output_parser=StrOutputParser()
    )

    return chain

# Function to add message to chat history
def add_message_to_history(role, content):
    chat_history = chat_history_var.get()
    chat_history.append({"role": role, "content": content})
    chat_history_var.set(chat_history)
