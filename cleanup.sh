#!/bin/bash

# containers
echo "Stopping and removing containers..."
docker rm -f nnfuzz_flask nnfuzz_tensorfuzz nnfuzz_deephunter nnfuzz_dlfuzz 2>/dev/null

# volume
echo "Removing the shared volume..."
docker volume rm nnfuzz_shared 2>/dev/null

# images and everything else
docker system prune -a

echo "Cleanup completed. All nnfuzz containers and volumes have been removed."
