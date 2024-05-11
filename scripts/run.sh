#!/bin/bash

# Set the name of your Docker image
IMAGE_NAME="video_outlier_optimization"
IMAGE_PATH="./container/Dockerfile"

# Build the Docker image
docker build -t $IMAGE_NAME -f $IMAGE_PATH .

# Run the Docker container and execute the command
docker run -p 8888:8888 $IMAGE_NAME

