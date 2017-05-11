# flower-recognition

use tensorflow inception

## Retrain models/inception

1. `INCEPTION_MODEL_DIR=$HOME/work/temp/inception-v3-model`
2. `mkdir -p ${INCEPTION_MODEL_DIR}`
3. `cd ${INCEPTION_MODEL_DIR}`
4. `curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gzcurl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz`
5. `tar xzf inception-v3-2016-03-01.tar.gz`
6. `MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"`
7. `TRAIN_DIR=$HOME/work/flower-recognition/dataset`
8. `VALIDATION_DIR=$HOME/work/flower-recognition/dataset2`
9. In particular, you will need to create a directory of training images that reside within $TRAIN_DIR and $VALIDATION_DIR arranged as such:

  ```$train_dir/dog/image0.jpeg
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

  ```
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

10. `FLOWERS_DATA_DIR=$HOME/work/flower-recognition/final_data/`
11. `TRAIN_DIR=$HOME/work/temp/flowers_train/`
12. `MODEL_PATH="${INCEPTION_MODEL_DIR}/inception-v3/model.ckpt-157585"`
12. retrain the dataset

  ```
  python3 ~/work/models/inception/inception/flowers_train.py --train_dir="${TRAIN_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1
  ```

## Retrain tensorflow in docker

[the tensorflow for poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#1)
