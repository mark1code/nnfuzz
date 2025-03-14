# nnfuzz

A web-based platform to automatically fuzz neural network models using tools TensorFuzz, DL Fuzz, and DeepHunter.

# How to Build and Run

Quickly build and run the entire setup using the `setup.sh` script. Make sure the script is executable:

```bash
chmod +x setup.sh
./setup.sh
```

This will handle removing existing containers, recreating the shared volume, building all components, and starting the Flask container.

# Access the Web Interface

Once the Flask container is running, navigate to `http://localhost:5000` in your web browser to access the nnfuzz web interface.

---

## Individual commands (for manual testing)

Remove existing containers (if any):
```bash
docker rm -f nnfuzz_flask nnfuzz_tensorfuzz
```

Remove and recreate the shared volume:
```bash
docker volume rm nnfuzz_shared
docker volume create nnfuzz_shared
```

Build and run the Flask container:
```bash
docker build -t nnfuzz_flask -f app/Dockerfile app
docker run -d -p 5000:5000 --name nnfuzz_flask \
    -v nnfuzz_shared:/shared \
    -v /var/run/docker.sock:/var/run/docker.sock \
    nnfuzz_flask
```

Build the TensorFuzz container:
```bash
docker build -t nnfuzz_tensorfuzz -f tools/tensorfuzz/Dockerfile tools/tensorfuzz
```

Build the DeepHunter container:
```bash
docker build -t nnfuzz_deephunter -f tools/deephunter/Dockerfile tools/deephunter
```

Build the DLFuzz container:
```bash
docker build -t nnfuzz_dlfuzz -f tools/dlfuzz/Dockerfile tools/dlfuzz
```

---

