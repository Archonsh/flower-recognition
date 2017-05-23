# flower-recognition

use tensorflow inception

## Retrain models/inception

### Reformat flower-data

1. `INCEPTION_MODEL_DIR=$HOME/work/temp/inception-v3-model`
2. `mkdir -p ${INCEPTION_MODEL_DIR}`
3. `cd ${INCEPTION_MODEL_DIR}`
4. `curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gzcurl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz`
5. `tar xzf inception-v3-2016-03-01.tar.gz`
6. `MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"`
7. `TRAIN_DIR=$HOME/work/flower-recognition/dataset`
8. `VALIDATION_DIR=$HOME/work/flower-recognition/dataset2`
9. In particular, you will need to create a directory of training images that reside within $TRAIN_DIR and $VALIDATION_DIR arranged as such:

  ```bash
  $train_dir/dog/image0.jpeg
  $TRAIN_DIR/dog/image1.jpg
  $TRAIN_DIR/dog/image2.png
  ...
  $TRAIN_DIR/cat/weird-image.jpeg
  $TRAIN_DIR/cat/my-image.jpeg
  $TRAIN_DIR/cat/my-image.JPG
  ...
  $VALIDATION_DIR/dog/imageA.jpeg
  $VALIDATION_DIR/dog/imageB.jpg
  $VALIDATION_DIR/dog/imageC.png
  ...
  $VALIDATION_DIR/cat/weird-image.PNG
  $VALIDATION_DIR/cat/that-image.jpg
  $VALIDATION_DIR/cat/cat.JPG
  ...
  ```

  ```bash
  OUTPUT_DIRECTORY=$HOME/work/flower-recognition/final_data/
  python3 ~/work/models/inception/inception/data/build_image_data.py \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
  ```

### Retrain the model

1. `FLOWERS_DATA_DIR=$HOME/work/flower-recognition/final_data/`
2. `TRAIN_DIR=$HOME/work/temp/flowers_train/`
3. `MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"`
4. retrain the dataset

  ```bash
  python3 ~/work/models/inception/inception/flowers_train.py --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
  ```

### Problems

最后会发现我的电脑上会不断的跑，不停地训练，具体我也不知道是什么原因，但是至少目前还没有找到解决办法，所以只能等以后再说了。没有 GPU 加持的电脑确实不太适合做深度学习相关的 tutorial ，以后有机会可以用 GPU 尝试一下，估计还是训练速度的问题。具体的任何操作与问题可以参考 [inception](https://github.com/tensorflow/models/tree/master/inception) ，关于这一种方法重新训练 Google Models 暂时就只研究到这里了，考虑下下面的 Docker 的方法。

## Retrain tensorflow in docker

### 前期安装工作

#### 安装 Docker 与测试

从 [docker](https://www.docker.com/community-edition) 官方网站下载对应的镜像文件安装就可以了，当安装完成之后，需要对 docker 是否正确运行进行检测，输入以下命令，如果没有报错那么说明一切顺利，可以开始进行下一步。

```bash
$ docker run hello-world
```

#### 在 Docker 中安装 Tensorflow 并测试

Docker 并不自带 Tensorflow 包，因此需要手动从 Docker 镜像源上下载，输入以下命令：

```bash
$ docker run -it tensorflow/tensorflow:1.1.0 bash
```

如果之前没有安装过 Tensorflow，就会自动从官网上下载，如下所示：

```bash
Unable to find image 'tensorflow/tensorflow:1.1.0' locally
1.1.0: Pulling from tensorflow/tensorflow
c62795f78da9: Pull complete
d4fceeeb758e: Pull complete
5c9125a401ae: Pull complete
0062f774e994: Pull complete
6b33fd031fac: Pull complete
353b34ef0a98: Pull complete
4f6aefc14b68: Pull complete
ce066374c6ca: Pull complete
c0755a91ab3a: Pull complete
f03279b52d25: Pull complete
d1c27c29b7e3: Pull complete
23807c5f4b3e: Pull complete
Digest: sha256:27bd43f1cf71c45eb48cb6e067b7cef47b168ac11c685d55a3495d27f0d59543
Status: Downloaded newer image for tensorflow/tensorflow:1.1.0
```

最后进入下面这个状态说明安装成功。

```bash
root@38038715aa2f:/notebooks#
```

输入 python，进入 python 命令行编辑界面，输入以下命令：

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session() # It will print some warnings here.
print(sess.run(hello))
```

如果系统正确输出了 "Hello TensorFlow!"，则说明整个 Docker 以及 Tensorflow 运行正常，可以开始进行下一步的操作了。'Ctrl-d'离开 Docker



[the tensorflow for poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#1)
