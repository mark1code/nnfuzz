FROM python:3.6-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy local tensorfuzz code
COPY . /app/tensorfuzz

WORKDIR /app

# Install required Python dependencies
RUN pip install \
    numpy==1.16.4 \
    absl-py==0.9.0 \
    tensorflow-gpu==1.15.0 \
    scipy==1.2.3 \
    pyflann-py3 \
    matplotlib

# Install debugging tools
RUN pip install ipdb

# Set environment variable for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2 

# Set the Python path to include the tensorfuzz code
ENV PYTHONPATH "${PYTHONPATH}:/app/tensorfuzz"

# Mount the shared volume
VOLUME /app/shared

# Default command to run fuzz_runner.py
CMD ["python", "fuzz_runner.py"]