# DLFuzz Dockerfile
FROM python:2.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy DL Fuzz repository and fuzzer runner
COPY DLFuzz /app/dlfuzz
COPY dlfuzz_fuzzer.py /app/dlfuzz_fuzzer.py

# Change the working directory to the DL Fuzz repository
WORKDIR /app/dlfuzz

# Install required Python dependencies for DL Fuzz
RUN pip install \
    tensorflow==1.15.0 \
    Keras==2.1.3 \
    h5py==2.7.1 \
    Pillow==5.0.0 \
    opencv-python==3.4.0.12

# Set Python path correctly
ENV PYTHONPATH "${PYTHONPATH}:/app/dlfuzz"

# Default command
CMD ["python", "/app/dlfuzz_fuzzer.py"]
