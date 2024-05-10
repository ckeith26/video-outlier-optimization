#!/bin/bash

# Set the name of your Docker image
IMAGE_NAME="../container/Dockerfile"

# Build the Docker image (if it hasn't been built yet)
docker build -t $IMAGE_NAME .

# Run the Docker container and execute the command
docker run --rm -it $IMAGE_NAME
