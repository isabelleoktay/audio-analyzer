#!/bin/bash

# Activate your conda environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Start Python service in background
echo "Starting Python service..."
cd python-service
conda run -n tf-from-source flask run --no-reload --host=0.0.0.0 --port=8080 > python-service.log 2>&1 &
PYTHON_PID=$!

# Wait until Python service responds on port 8080
echo "Waiting for Python service to be ready..."
until curl -s http://localhost:8080/python-service/ > /dev/null; do
  sleep 1
done

echo "Python service is ready!"

# Go back to root directory
cd ..

# Start backend and frontend (can run concurrently)
echo "Starting backend and frontend..."
npm run backend &
npm run frontend &

wait $PYTHON_PID
