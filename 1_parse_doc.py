import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pathlib import Path

# Allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize the parser with multimodal settings
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY_3"),
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    parsing_instruction = """
                            This document presents comprehensive information on Peugeot's range of electric vehicles, including technical specifications, charging details, cost information and tax incentives, as well as associated services.
                            It contains numerous structured sections, technical data tables, and information on different electric vehicle models.
                            The document also covers topics such as vehicle range, battery technologies, connected mobility services, and the Peugeot Allure Care warranty program.
                            It includes precise numerical data on vehicle performance, costs, and government incentives.Also at the end of each paragraph please add "<>end_paragraph<>" to separate the paragraphs.    
                            """,

    result_type="text",
    

)

# Use SimpleDirectoryReader to parse the file
file_extractor = {".pdf": parser}

try:
    raw_path = Path(__file__).parent
    raw_path = raw_path / "raw_data"
    
    documents = SimpleDirectoryReader(raw_path, file_extractor=file_extractor).load_data()


    # Save the parsed result to a file
    with open('peugeot_data.txt', 'w') as result_file:
        for doc in documents:
            result_file.write(doc.text)
except Exception as e:
    print(f"Error while parsing the file: {e}")
    
