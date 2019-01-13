```shell
docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0 \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.15.0' \
  --build-arg TF_BUILD_BRANCH='r1.12' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0 && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.5.0 && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.12.0_dali0.6.0

```

```shell
docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0 \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.19.2' \
  --build-arg TF_BUILD_BRANCH='r1.13' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0 && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0_dali0.5.0 && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tf1.13.0_dali0.6.0

```

```shell
docker build -t born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster \
  --build-arg BASE_CONTAINER='cudnn_openmpi' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1' \
  --build-arg BAZEL_VERSION='0.19.2' \
  --build-arg TF_BUILD_BRANCH='master' \
  --build-arg  TF_REPO='https://github.com/tensorflow/tensorflow.git' \
  --build-arg HOROVOD_VERSION='0.15.2' \
  - < Dockerfile.tensorflow && \
docker push born2data/tensorflow:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.5.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster' \
  --build-arg DALI_BUILD_BRANCH='v0.5.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.5.0 && \
docker build -t born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.6.0  \
  --build-arg BASE_CONTAINER='tensorflow' \
  --build-arg BASE_CONTAINER_TAG='ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster' \
  --build-arg DALI_BUILD_BRANCH='v0.6.0' \
  --build-arg BUILD_DALI_FOR_TF='ON' \
  - < Dockerfile.dali && \
docker push born2data/tensorflow_dali:ubuntu16.04_cuda10.0_cudnn7.4.2_openmpi3.4.1_hvd0.15.2_tfmaster_dali0.6.0

```