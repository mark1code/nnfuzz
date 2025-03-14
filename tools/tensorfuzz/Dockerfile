# TensorFuzz Dockerfile
FROM python:3.6-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# copy tensorfuzz repository and fuzzer runner
COPY tensorfuzz /app/tensorfuzz
COPY tensorfuzz_fuzzer.py /app/tensorfuzz_fuzzer.py

WORKDIR /app

RUN pip install \
    numpy==1.16.4 \
    absl-py==0.9.0 \
    tensorflow==1.15.0 \
    scipy==1.2.3 \
    pyflann-py3 \
    matplotlib \
    h5py==2.10.0

ENV PYTHONPATH "${PYTHONPATH}:/app/tensorfuzz"

# Default command
CMD ["python", "/app/tensorfuzz_fuzzer.py"]
