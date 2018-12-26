# GPU_DockerFiles
Various DockerFiles based for Deep Learning using Nvidia-Docker

## 1. Build the base image

### 1.1 Using the NVIDIA CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn:ubuntu16.04_cuda10.0_cudnn7.4.2.svg)

```shell
docker build -t born2data/cudnn:ubuntu16.04_cuda10.0_cudnn7.4.2 \
  --build-arg BASE_CONTAINER='cuda' \
  --build-arg CUDA_BASE="10.0" \
  --build-arg CUDNN_VERSION="7.4.2.24-1" \
  --build-arg NCCL_VERSION="2.3.7-1" \
  --build-arg LIBNVINFER_VERSION="5.0.2-1" \
  - < Dockerfile.base && \
docker push born2data/cudnn:ubuntu16.04_cuda10.0_cudnn7.4.2
```

### 1.2 Using the NVIDIA CUDAGL (CUDA + OpenGL) Container => Useful to display X11 windows ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_gl:ubuntu16.04_cuda10.0_cudnn7.4.2.svg)

```shell
docker build -t born2data/cudnn_gl:ubuntu16.04_cuda10.0_cudnn7.4.2 \
  --build-arg BASE_CONTAINER='cudagl' \
  --build-arg CUDA_BASE="10.0" \
  --build-arg CUDNN_VERSION="7.4.2.24-1" \
  --build-arg NCCL_VERSION="2.3.7-1" \
  --build-arg LIBNVINFER_VERSION="5.0.2-1" \
  - < Dockerfile.base && \
docker push born2data/cudnn_gl:ubuntu16.04_cuda10.0_cudnn7.4.2
```

## 2. Add OpenMPI on top of the previous container to enable multi GPU training / inference

### 2.1 From CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_openmpi:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1.svg)

```shell
docker build -t born2data/cudnn_openmpi:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1 \
  --build-arg BASE_CONTAINER=cudnn \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2' \
  --build-arg OPENMPI_VERSION="3.1.2" \
  - < Dockerfile.openmpi && \
docker push born2data/cudnn_openmpi:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1
```

### 2.2 From CUDAGL Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1.svg)

```shell
docker build -t born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1 \
  --build-arg BASE_CONTAINER=cudnn_gl \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2' \
  --build-arg OPENMPI_VERSION="3.1.2" \
  - < Dockerfile.openmpi && \
docker push born2data/cudnn_openmpi_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1
```

## 3. Install your favorite Deep Learning Framework with Horovod for easy multi GPU inference/training.

### 3.1 Build a Tensorflow Container with Horovod Pre-Packaged on top of container built in 2)

### 3.1.1 From CUDA Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0.svg)

```shell
docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0 \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.15.0' \
  --build-arg TF_BUILD_BRANCH='r1.12' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0


docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13 \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.19.2' \
  --build-arg TF_BUILD_BRANCH='r1.13' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13


docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.19.2' \
  --build-arg TF_BUILD_BRANCH='master' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster


docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.19.2' \
  --build-arg TF_BUILD_BRANCH='XLADeviceStreams' \
  --build-arg TF_REPO='https://github.com/DEKHTIARJonathan/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix
```

### 3.1.2 From CUDAGL Container ![Docker Badge](https://images.microbadger.com/badges/image/born2data/tensorflow_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0.svg)

```shell
docker build -t born2data/tensorflow_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0  \
  --build-arg BASE_CONTAINER='cudnn_openmpi_gl' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.15.0' \
  --build-arg TF_BUILD_BRANCH='r1.12' \
  --build-arg TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0
```

### 3.2 Build a PyTorch Container with Horovod Pre-Packaged on top of container built in 2)
**This feature is under development and not available at the moment**

### 3.3 Build a MxNet Container with Horovod Pre-Packaged on top of container built in 2)
**This feature is under development and not available at the moment**

# 4. Installing NVIDIA DALI for optimal performances

## 4.1 On top of Tensorflow

### 4.1.1 DALI v0.5.0

```shell

docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13_dali0.5.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.5.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix_dali0.5.0


docker build -t born2data/tensorflow_dali_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow_gl' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0
```

### 4.1.2 DALI v0.6.0

```
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfr1.13_dali0.6.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.6.0


docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix_dali0.6.0


docker build -t born2data/tensorflow_dali_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow_gl' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0
```

# 5. Launching the images

## 5.1 Tensorflow image

```shell
nvidia-docker run -it --rm \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/:/workspace/ \
    born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0
```

```shell
nvidia-docker run -it --rm \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/:/workspace/ \
    -v /mnt/dldata/imagenet/val-jpeg:/data/imagenet-img/val/ \
    born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf_xla_fix_dali0.5.0
```

## 5.2 Tensorflow + OpenGL image

```shell
nvidia-docker run -it --rm \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    -v $(pwd)/:/workspace/ \
    born2data/tensorflow_gl:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0
```
