import wget
import zipfile
import glob2
import os
from time import sleep

url = 'https://test-backet-dsti.s3.amazonaws.com/kagglecatsanddogs_3367a.zip'
pathdir = './AWS/model_deploy/classification/data/raw'

def download_zip(pathdir, url):
  if os.path.exists(pathdir):

    print('Downloading from %s, this may take a while...' % url)
    #myfile = wget.download(url, pathdir)
    print("")
    #os.path.join(pathdir, myfile)
    #zip_file = wget.download(url)
    #wget.download(url, pathdir)
    myfile = os.path.join(path, x)
    #file = glob2.glob(f'{pathdir}/*.zip')
    #myfile = glob(f"{pathdir}/*.zip")
    #with zipfile.ZipFile(myfile, 'r') as zip_ref:
    #  zip_ref.extractall(pathdir)  

    print(myfile)

if __name__=='__main__':
  print("Downloading starts ...")
  download_zip(pathdir, url)
  print('Zip done !')

#client= mlflow.tracking.MlflowClient("ec2-52-215-94-6.eu-west-1.compute.amazonaws.com:5000")
#client= mlflow.tracking.MlflowClient("ec2-52-215-94-6.eu-west-1.compute.amazonaws.com:5000")
# TRACKING_URI = "http://52.215.94.6:5000/"
#mlflow.tracking.set_tracking_uri(TRACKING_URI)
#mlflow.start_run()
#mlflow.log_metric("score", 94)
# zip model and copy it s3 bucket
# aws s3 cp model_ec2.zip s3://test-backet-dsti/ --region us-east-1
# remove the data after
