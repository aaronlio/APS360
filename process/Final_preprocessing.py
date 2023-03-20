import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import pandas as pd
import PIL.Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt


def fetch(url):
  image = PIL.Image.open(requests.get(url, stream=True, timeout = 1.5).raw)
  return image

def fetch_and_show(url):
  fetch(url).show()

def fetch_show_with_caption(data, index=0, url_col_name='image_url', caption_col_name='caption'):
  fetch_and_show(data[url_col_name].iloc[index])
  print(data[caption_col_name].iloc[index])


def transform_to_tensor(obj):
  if isinstance(obj, PIL.JpegImagePlugin.JpegImageFile):
    return transform(obj)
  
  elif type(obj) == str:
    try:
      transformed = transform(fetch(obj))
    except:
      raise Exception("Argument must be a url or PIL image object")

    return transformed

if __name__ == '__main__':

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.PILToTensor()])


    raw_data = pd.read_csv('/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/Train_GCC-training.tsv', sep='\t', names=["caption", "image_url"])

    transform_to_tensor(fetch(raw_data['image_url'].iloc[0]))

    raw_data = raw_data.iloc[0:25000]

    image_tensors = torch.zeros((1,3,224,224))
    i = 0
    url_list = raw_data['image_url'].values.tolist()
    for url in url_list:
        try:
            image_tensors = torch.cat((image_tensors, transform_to_tensor(url).unsqueeze(0)))
        except Exception as e:
            raw_data = raw_data.drop(raw_data[raw_data['image_url']==url].index.values)
        else:
            i+=1
            print(i)

        
    torch.save(image_tensors, '/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/process/big_data.pt')
    raw_data.to_csv('/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/process/big_data_labels.tsv', sep='\t')

    print("Data Saved!")

    images = image_tensors[1:,:,:,:]
    #print(images[1].shape)

    #plt.imshow(  (images[1]*255).permute(1, 2 , 0).astype(np.uint8)  )


    data = raw_data.drop(columns='image_url')

    #print(images.shape)
    #plt.imshow(  images[5000].permute(1, 2 , 0)  )
    #plt.figtext(0.5, 0.01, data['caption'].iloc[5000], wrap=True, horizontalalignment='center', fontsize=12)
    #plt.show()

    labels = pd.concat([data]*4, ignore_index=True)

    transform0 = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transform1 = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomCrop(71)
    ])


    transform2 = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    transform3 = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomApply(torch.nn.ModuleList([transforms.Grayscale(3),transforms.GaussianBlur(3), transforms.RandomInvert(p=1), transforms.RandomRotation(degrees=(0,180))]))
        ])

    transformed0 = transform0(images)
    print("1")
    transformed1 = transform1(images)
    print("2")
    transformed2 = transform2(images)
    print("3")
    transformed3 = transform3(images)
    print("4")
    augmented_images = torch.cat([transformed0, transformed1, transformed2, transformed3], dim=0)

    print(augmented_images.shape, labels.shape)

    torch.save(augmented_images, '/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/process/big_aug_images.pt')

    labels.to_csv('/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/process/big_aug_labels.tsv', sep='\t', columns=["caption"])