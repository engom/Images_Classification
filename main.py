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
import boto3
import mlflow  # from mlflow import log_metric, log_param, log_artifacts

# creation a session to connect to s3
session = boto3.Session(
                        aws_access_key_id=secrets.AWSACCESSKEYID,
                        aws_secret_access_key=secrets.AWSSECRETKEY,
                        region_name='us-east-1'
                        )

s3 = session.resource('s3')
# specify the bucket_name
my_bucket = s3.Bucket('test-backet-dsti')

# import codes
import src.data.data_loader as loader
import src.models.cnn as modeler
import src.data.processing_data as processor
# link of source data
src = 'https://test-backet-dsti.s3.amazonaws.com/kagglecatsanddogs_3367a.zip'
# path of destination director
dst = '/home/ubuntu/project/classification/data/raw'

# main program starts
if __name__=='__main__':
  # load and unzip
  loader.download_zip(dst, src)
  # data processing
  train_ds, val_ds = processor.data_processor(dst)
  # build model
  image_size = (180, 180)
  batch_size = 32
  model = modeler.make_model(input_shape=image_size + (3,), num_classes=2) 
  # train model
  epochs = 5
  callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), ]
  model.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"],)
  model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=4)
  # save model to s3
  my_bucket.upload_file(model.save('model.h5'))
  
  # Track the model with mlflow
  TRACKING_URI = "http://44.198.184.229:5000/"
  mlflow.tracking.set_tracking_uri(TRACKING_URI)
  mlflow.start_run()
