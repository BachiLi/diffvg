# Use the official Python image as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install the project
RUN python setup.py install

# Set the entry point to run the project on Hugging Face
ENTRYPOINT ["python", "single_circle.py"]
