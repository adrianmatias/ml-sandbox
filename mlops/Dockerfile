# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . /app

# Make the serve script executable
RUN chmod +x serve

# Expose the port the app runs on
EXPOSE 8080

# Define the entry point script
ENTRYPOINT ["./serve"]