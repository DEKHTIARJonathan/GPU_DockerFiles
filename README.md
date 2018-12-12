# GPU_DockerFiles
Various DockerFiles based for Deep Learning using Nvidia-Docker

## 1. Build the base image

### 1.1 Using the NVIDIA CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn:ubuntu16.04_cuda10.0.130_cudnn7.4.1.svg)

```
docker build -t born2data/cudnn:ubuntu16.04_cuda10.0.130_cudnn7.4.1 \
  --build-arg repository=cuda \
  - < Dockerfile.base

docker push born2data/cudnn:ubuntu16.04_cuda10.0.130_cudnn7.4.1
```

### 1.2 Using the NVIDIA CUDAGL (CUDA + OpenGL) Container => Useful to display X11 windows ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1.svg)

```
docker build -t born2data/cudnn_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1 \
  --build-arg repository=cudagl \
  - < Dockerfile.base

docker push born2data/cudnn_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1
```

## 2. Add OpenMPI on top of the previous container to enable multi GPU training / inference

### 2.1 From CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_openmpi:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1.svg)

```
docker build -t born2data/cudnn_openmpi:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1 \
  --build-arg repository=cudnn \
  - < Dockerfile.openmpi

docker push born2data/cudnn_openmpi:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1
```

### 2.2 From CUDAGL Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1.svg)

```
docker build -t born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1 \
  --build-arg repository=cudnn_gl \
  - < Dockerfile.openmpi

docker push born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1
```

## 3. Install your favorite Deep Learning Framework with Horovod for easy multi GPU inference/training.

### 3.1 Build a Tensorflow Container with Horovod Pre-Packaged on top of container built in 2)

### 3.1.1 From CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/tensorflow:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0.svg)

```
docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0 \
  --build-arg repository=cudnn_openmpi \
  - < Dockerfile.tensorflow

docker push born2data/tensorflow:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0
```

### 3.1.2 From CUDAGL Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/tensorflow_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0.svg)

```
docker build -t born2data/tensorflow_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0  \
  --build-arg repository=cudnn_openmpi_gl \
  - < Dockerfile.tensorflow

docker push born2data/tensorflow_gl:ubuntu16.04_cuda10.0.130_cudnn7.4.1_openmpi3.4.1_hvd0.15.2_tf1.12.0
```

### 3.2 Build a PyTorch Container with Horovod Pre-Packaged on top of container built in 2)
**This feature is under development and not available at the moment**

### 3.3 Build a MxNet Container with Horovod Pre-Packaged on top of container built in 2)
**This feature is under development and not available at the moment**
