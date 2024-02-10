# Set base image (host OS)  
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV PATH=/usr/local/cuda/:${PATH} 
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH  

# Update system packages and install build dependencies
RUN apt-get update && \      
    apt-get upgrade -y && \    
    apt-get install -y git build-essential  && \    
    apt-get clean    

RUN pip install packaging wheel

# Install unsloth training framework
RUN pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton  \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install "unsloth[cu121_ampere] @ git+https://github.com/unslothai/unsloth.git"  

# Install project-specific dependencies    
COPY ./requirements/laser.txt laser.txt
RUN python -m pip install -r laser.txt

# Install Mixtral MergeKit
RUN git clone --branch mixtral https://github.com/cg123/mergekit.git && \
    cd mergekit && pip install -e .    

COPY ./requirements/app.txt app.txt
RUN python -m pip install -r app.txt
# Cleanup step  
RUN apt-get remove -y git build-essential && \  
    apt-get autoremove -y && \  
    apt-get clean && \  
    rm -rf /var/lib/apt/lists/*

# Application setup
WORKDIR /app
COPY api ./api
RUN mkdir -p models/llm && \
    mkdir -p models/emb && \
    mkdir -p models/rr

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "5"]  
