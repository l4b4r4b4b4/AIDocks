# TODO Can I build application but later run it with runtime base image 2.1.2-cuda12.1-cudnn8-runtime ?
# Set base image (host OS)  
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update && \      
    apt-get upgrade -y && \    
    apt-get install -y git build-essential  && \    
    apt-get clean    

RUN pip install packaging wheel

# Install project-specific dependencies    
COPY ./requirements ./requirements
RUN cd requirements && \
    python -m pip install -r laser.txt  && \     
    python -m pip install -r app.txt

# Install unsloth training framework
RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton  \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install "unsloth[cu121_ampere] @ git+https://github.com/unslothai/unsloth.git"  

# Install Mixtral MergeKit
RUN git clone --branch mixtral https://github.com/cg123/mergekit.git && \
    cd mergekit && pip install -e .    

# Cleanup step  
RUN apt-get remove -y git build-essential && \  
    apt-get autoremove -y && \  
    apt-get clean && \  
    rm -rf /var/lib/apt/lists/*

# App Config
WORKDIR /app

# Set up application
COPY api ./api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]  
