FROM --platform=linux/amd64 python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy everything from the current directory to /app in the container
COPY . /app/

# Install system dependencies
RUN apt update -y && apt install -y awscli curl

# Install Miniconda to manage dependencies from environment.yml
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create a Conda environment and install dependencies
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "germany_apartment_rent_prediction", "/bin/bash", "-c"]

# Set default command to run application
CMD ["conda", "run", "--no-capture-output", "-n", "germany_apartment_rent_prediction", "python", "application.py"]