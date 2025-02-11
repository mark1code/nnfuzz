FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc
COPY requirements.txt .
RUN pip install -r requirements.txt

# shared directory exists in the container
RUN mkdir -p /app/shared/uploads /app/shared/results

ENV DOCKER_HOST=unix:///var/run/docker.sock

COPY . .

# mount shared volume
VOLUME /app/shared

CMD ["python", "app.py"]
