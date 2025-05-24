# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Ensure langgraph, langchain, fastapi, uvicorn, and any other dependencies are listed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY agentworkflow.py agentworkflow.py
COPY service.py service.py

# Make port 80 available to the world outside this container
# FastAPI default port is 8000 for uvicorn, ensure this matches your uvicorn command
EXPOSE 8000

# Define environment variables (if any)
# ENV OPENAI_API_KEY="your_api_key_here" # Example, better to use secrets management

# Run uvicorn server when the container launches
# The command should match how you run uvicorn
# 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "service:api", "--host", "0.0.0.0", "--port", "8000"]