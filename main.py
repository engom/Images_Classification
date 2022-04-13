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
# import matplotlib.pyplot as plt
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
#<<<<<<< HEAD
dst = './data/raw'
=======
#dst = '/home/ubuntu/project/classification/data/raw'
#>>>>>>> ca356fbd7375eb98181dfc758105a03f04f83b17

def push_mlflow(TRACKING_URI, metric_name, value, run_name = 'Default'):
    """
    Function to send information to a given mlflow client.
    Args:
    TRACKING_URI (str): IP address info of the client (http://52.215.94.6:5000/)
    metric_name (str): Name of the metric to save
    value (float): Value of the metric
    run_name (str): Name of the run
    Returns:
    None
    """
    mlflow.tracking.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(run_name = run_name) as run:
        mlflow.log_metric(metric_name, value)
# Track the model with mlflow
TRACKING_URI = "http://44.198.184.229:5000/"
        
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
  
  #keras.utils.plot_model(model, show_shapes=True)
  
  ## mlflow 
  run_name = 'Default'
  metric_name='accuracy'
  value = 10

  #push_mlflow(TRACKING_URI, metric_name, value ,run_name = run_name)

  ## SNS
  client = boto3.client('sns', region_name='us-east-1')

  response = client.publish(
                TopicArn='arn:aws:sns:eu-west-2:321132792081:AWS_Topic',
                Message='Training done',
                Subject='Classification training'
                )
   print(response)
