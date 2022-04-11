import wget
import zipfile
from glob2 import glob
# import os
url = 'https://test-backet-dsti.s3.amazonaws.com/kagglecatsanddogs_3367a.zip'
pathdir = '/content/classication/data/raw'

def download_zip(pathdir, url):
  wget.download(url, pathdir)
  myfile = glob(f'{pathdir}/*.zip')[0]
  with zipfile.ZipFile(myfile, 'r') as zip_ref:
    zip_ref.extractall(pathdir)  

if __name__=='__main__':
  print("Downloading starts ...")
  download_zip(pathdir, url)
  print('Zip done !')