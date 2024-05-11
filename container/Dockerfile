# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY ./container/requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the required packages
RUN pip install -r requirements.txt

# Install the Jupyter kernel
RUN python -m ipykernel install --name my_kernel

# Copy the rest of the application code
COPY . .

# Set the entry point (if needed)
CMD ["python", "./scripts/setup.py"]
