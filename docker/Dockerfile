# Use Ubuntu 20.04 as base image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    bzip2 \
    software-properties-common \
    build-essential \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set environment variables for conda
ENV PATH="/opt/conda/bin:$PATH"

# Create the conda environment with Python and numpy
RUN conda create -n ps_baseline python=3.8.0 numpy=1.23.5 -y

# Activate conda environment for all following commands
SHELL ["/bin/bash", "-c"]

# Activate conda environment and install packages
RUN source activate ps_baseline && \
    pip install --upgrade pip==22.3.1 && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install soundfile==0.11.0 librosa==0.9.1 omegaconf==2.0.6 && \
    pip install conformer && \
    pip install scikit-learn pandas==1.3.5

# Clone the repository and install Fairseq
RUN git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git && \
    cd SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 && \
    source activate ps_baseline && pip install --editable ./ && \
    cd ../..

COPY . /app

# Set the default shell to activate the conda environment
CMD ["/bin/bash", "-c", "source activate ps_baseline && python monitor_python.py"]

