# STEPS
# download data from S3
# zip data
# process data
# split data
# build deep learning model
# train model
# evaluate model
# predict & inference
# Imports libraries
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import os

# import codes
import src.data.data_loader as loader
import src.models.cnn as modeler

# link of source data
src = 'https://test-backet-dsti.s3.amazonaws.com/kagglecatsanddogs_3367a.zip'
# path of destination director
dti = '/content/classication/data/raw'

# main program starts
if __name__=='__main__':
  # load and unzip
  loader.download_zip(dti, src)
  # data processing
  num_skipped = 0
  for folder_name in ("Cat", "Dog"):
      folder_path = os.path.join(f"{dti}/PetImages", folder_name)
      for fname in os.listdir(folder_path):
          fpath = os.path.join(folder_path, fname)
          try:
              fobj = open(fpath, "rb")
              is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
          finally:
              fobj.close()

          if not is_jfif:
              num_skipped += 1
              # Delete corrupted image
              os.remove(fpath)

  print("Deleted %d images" % num_skipped)

  image_size = (180, 180)
  batch_size = 32
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                f"{dti}/PetImages",
                                                                validation_split=0.2,
                                                                subset="training",
                                                                seed=1337, 
                                                                image_size=image_size,
                                                                batch_size=batch_size,
                                                                )
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                              f"{dti}/PetImages",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              seed=1337,
                                                              image_size=image_size,
                                                              batch_size=batch_size,
                                                              )
  # build model
  model = modeler.make_model(input_shape=image_size + (3,), num_classes=2) 
  # train model
  epochs = 50
  callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), ]
  model.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"],)
  model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=2)
