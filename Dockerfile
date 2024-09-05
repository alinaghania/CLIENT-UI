# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Ensure the fastapi-llm directory is in the Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set the working directory to the fastapi-llm directory
WORKDIR /app/fastapi-llm

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]