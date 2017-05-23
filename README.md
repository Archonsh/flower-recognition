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

### 安装 Docker 与测试

从 [docker](https://www.docker.com/community-edition) 官方网站下载对应的镜像文件安装就可以了，当安装完成之后，需要对 docker 是否正确运行进行检测，输入以下命令，如果没有报错那么说明一切顺利，可以开始进行下一步。

```bash
$ docker run hello-world
```

### 在 Docker 中安装 Tensorflow 并测试

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

### 创建文件夹及运行

在需要运行本程序的地方创建一个文件夹，用于和 Docker 内核进行数据交互，方便进行训练，然后输入以下代码，重新打开 Docker：

```bash
$ docker run -it \
  --publish 6006:6006 \
  --volume ${HOME}/work/flower-recognition/tf_files:/tf_files \
  --workdir /tf_files \
  tensorflow/tensorflow:1.1.0 bash
```

其中 tf_files 是创建的文件夹的具体位置，运行正常的话会收到如 `root@xxxxxxxxx:/tf_files#` 这样的提示。

### 获取图片数据集

这个可以从网络上自己直接获取你需要进行分类的种类的图片集合即可，我这里使用了杜鹃花的种类，这个可以直接在 Chrome 浏览器上应用插件批量获得谷歌提供的图片，当然如果需要用作商用的话尽量避开那些有版权的图片就可以了，如果不想自己动手麻烦的话，可以参考以下的官方提供的数据集：

```bash
$ curl -O http://download.tensorflow.org/example_images/flower_photos.tgz tar xzf flower_photos.tgz
$ ls flower_photos
```

增加训练效率，减少图片数量：

```bash
$ ls flower_photos/roses | wc -l
$ rm flower_photos/*/[3-9]*
$ ls flower_photos/roses | wc -l
```

### 进行自定义训练

首先需要从 Tensorflow 下载 inception 的训练脚本：

```bash
$ curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```

获得了脚本之后，将自定义的数据集放入 `/tf_files` 当中，然后打开 Tensorflow 监控器：

```bash
$ tensorboard --logdir training_summaries &
```

如果已经打开了，就可以用 `pkill -f "tensorboard"` 杀死即可，接下来开始正式的训练，设置各种参数配合脚本运行：

```bash
$ python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=dataset
```

如果遇到 `[Errno socket error] [Errno 101] Network is unreachable` 报错的话，那没办法了，开关多几次 VPN，使用不同的路线，网络不能连接到的话，都是属于玄学的问题，毕竟在国内有方教授的墙横着，大家都没什么办法。

如果遇到 `ERRO[XXXX] error getting events from daemon: EOF` 这种 bug 的话，需要配置你的 CPU 核数？具体我也不是很懂，因为没有遇到，详情可以参考 [issue](https://github.com/moby/moby/issues/31220)，应该是进行小配置即可。

一切顺利的话，你的terminal会出现下面的输出，接下来就只需要等待训练结束即可了。

```bash
2017-05-23 06:45:41.577876: Step 240: Validation accuracy = 89.0% (N=100)
2017-05-23 06:45:48.890883: Step 250: Train accuracy = 100.0%
2017-05-23 06:45:48.890964: Step 250: Cross entropy = 0.056585
```

如果你对你的机器非常有自信的话，CPU强大的话（至少不能像我这种），可以考虑采取准确率更高的方式进行训练，上述的例子训练步数才500，如果你没有在刚刚步骤当中减少对应的图片集的话，也就是说如果你有巨大的数据集需要处理的话，500步是显然不够的，需要使用默认的4000步就可以了，如：

```bash
$ python /tensorflow/tensorflow/examples/image_retraining/retrain.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=inception \
  --summaries_dir=training_summaries/long \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=dataset
```

关于CNN的具体知识就不在这里赘述了，以后有机会的话会详细在记录一下学习的过程，如果你的脚本训练顺利的话，会得到以下的结果：

```bash
Final test accuracy = 84.6% (N=13)
Converted 2 variables to const ops.
```

### 使用训练好的模型

训练好的模型会自动生成在`tf_files/retrained_graph.pb`，其中还有一些其他文件，比如样本标签文件`tf_files/retrained_labels.txt`等，有了这些文件之后就可以像[picture_recognition](https://www.tensorflow.org/versions/master/tutorials/image_recognition)上所写的一样了，PS：最初我也是从这里开始接触图片分类的。







[the tensorflow for poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#1)
