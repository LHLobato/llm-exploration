FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Desabilita verificação de GPG
RUN echo 'Acquire::AllowInsecureRepositories "true";' > /etc/apt/apt.conf.d/99insecure \
    && echo 'Acquire::AllowDowngradeToInsecureRepositories "true";' >> /etc/apt/apt.conf.d/99insecure \
    && echo 'APT::Get::AllowUnauthenticated "true";' >> /etc/apt/apt.conf.d/99insecure

# Dependências do sistema
RUN apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Symlinks simples sem update-alternatives
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# PyTorch com suporte a CUDA 12.4
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Libs do script de treino
RUN pip3 install \
    transformers \
    datasets \
    peft \
    bitsandbytes \
    accelerate \
    evaluate \
    scikit-learn \
    scipy \
    numpy \
    pandas \
    huggingface_hub \
    optuna \
    sentencepiece \
    protobuf


RUN pip3 install \
    rouge_score \
    sacrebleu \
    absl-py

CMD ["bash"]
