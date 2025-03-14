#!/bin/bash

echo "Removing existing containers..."
docker rm -f nnfuzz_flask nnfuzz_tensorfuzz nnfuzz_deephunter nnfuzz_dlfuzz 2>/dev/null

echo "Remaking shared volume..."
docker volume rm nnfuzz_shared 2>/dev/null
docker volume create nnfuzz_shared

# flask container
echo "Building and running the Flask container..."
docker build -t nnfuzz_flask -f app/Dockerfile app
docker run -d -p 5000:5000 --name nnfuzz_flask \
    -v nnfuzz_shared:/shared \
    -v /var/run/docker.sock:/var/run/docker.sock \
    nnfuzz_flask

# tensorfuzz container
echo "Building the TensorFuzz container..."
docker build -t nnfuzz_tensorfuzz -f tools/tensorfuzz/Dockerfile tools/tensorfuzz

# deephunter container
echo "Building the DeepHunter container..."
docker build -t nnfuzz_deephunter -f tools/deephunter/Dockerfile tools/deephunter

# dlfuzz container
echo "Building the DLFuzz container..."
docker build -t nnfuzz_dlfuzz -f tools/dlfuzz/Dockerfile tools/dlfuzz

echo "Checking if Flask container is running..."
docker ps | grep nnfuzz_flask

echo "Checking Flask container logs for errors..."
docker logs nnfuzz_flask

echo "All containers built and Flask server running. http://localhost:5000"