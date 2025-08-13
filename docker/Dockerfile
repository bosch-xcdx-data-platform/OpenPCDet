# CUDA 11.8 + cuDNN8, with build tools (nvcc) for compiling extensions
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (single Python: 3.8 on 20.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip python3-venv \
      git curl ca-certificates wget build-essential \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      bzip2 tree htop g++ && \
    rm -rf /var/lib/apt/lists/*

# Make python3/pip3 the defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

# --- IMPORTANT: avoid GPU probing at build time ---
# Choose the archs you need; examples:
# T4/V100(7.0), Turing 20xx (7.5), A100 (8.0), RTX 30xx/A10 (8.6)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# PyTorch/TV + cu118 wheels
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Core Python deps (keep NumPy modern for cu118 toolchains)
RUN pip install --no-cache-dir \
      numpy \
      tensorboardX easydict pyyaml scikit-image tqdm SharedArray six glob2 terminaltables \
      gdown

# Some packages people use with OpenPCDet (optional)
RUN pip install --no-cache-dir mayavi configobj

# Boost geometry headers (OpenPCDet uses)
RUN wget -q https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0* 

# spconv matching CUDA 11.8
RUN pip install --no-cache-dir spconv-cu118

# Copy your repo and build pcdet extensions
WORKDIR /app/OpenPCDet
COPY . /app/OpenPCDet

# (Optional) download model via gdown (Drive "uc?id=" links)
RUN mkdir -p /app/OpenPCDet/models && \
    gdown --id 19_jiAeGLz7V0wNjSJw4cKmMjdm5EW5By -O /app/OpenPCDet/models/voxel_rcnn_car.pth

# Build OpenPCDet (uses nvcc & PyTorch C++/CUDA extension)
# Thanks to TORCH_CUDA_ARCH_LIST, this won't try to probe a GPU.
# --- deps that must match Python 3.8 ---
    RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    numba==0.56.4 \
    llvmlite==0.39.0

RUN pip install --no-cache-dir --no-build-isolation -e .

#RUN python setup.py develop

# Convenience envs
ENV PYTHONPATH=$PYTHONPATH:/app/OpenPCDet

RUN apt-get update && apt-get install -y \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 && \
   rm -rf /var/lib/apt/lists/*

RUN pip pyquaternion opencv-python 
# Default command (override at run)

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyquaternion opencv-python-headless av2==0.2.1 numpy==1.23.5 numba==0.56.4 llvmlite==0.39.0 kornia==0.7.3

ENV MPLBACKEND=Agg
ENV PYTORCH_JIT=0
# todo: remove this in the next build
COPY . /app/OpenPCDet 
CMD ["bash"]