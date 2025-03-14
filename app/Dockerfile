FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install
COPY requirements.txt . 
RUN pip install -r requirements.txt

# Create shared directories
RUN mkdir -p /shared/uploads /shared/results

# Copy app code
COPY . .

# Set environment variables
ENV DOCKER_HOST=unix:///var/run/docker.sock
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose port and set command
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
