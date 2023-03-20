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
from zipfile import ZipFile

images = torch.load('/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/aug_images.pt').type(torch.uint8)

data = pd.read_csv('/Users/bossaaron3/Documents/Course Documents - 2nd Year/APS360/APS360_Project/aug_labels.tsv', sep='\t')

fig = plt.figure()
plt.title(data['caption'].iloc[2], wrap=False, horizontalalignment='center', fontsize=12)
plt.axis('off')
ax1 = fig.add_subplot(221)
im1 = ax1.imshow(  images[2].permute(1, 2 , 0))
ax1.set_xlabel('Unaugmented Image')
ax1.set_xticks([])
ax1.set_yticks([])


ax2 = fig.add_subplot(222)
im2 = ax2.imshow(  images[5703].permute(1, 2 , 0))
ax2.set_xlabel('Random Erased Image')
ax2.set_xticks([])
ax2.set_yticks([])


ax3 = fig.add_subplot(223)
im3 = ax3.imshow(  images[11404].permute(1, 2 , 0))
ax3.set_xlabel('Colour Jittered Image')
ax3.set_xticks([])
ax3.set_yticks([])


ax4 = fig.add_subplot(224)
im4 = ax4.imshow(  images[17105].permute(1, 2 , 0))
ax4.set_xlabel('Randomly Transformed Image')
ax4.set_xticks([])
ax4.set_yticks([])

#ax1.title.settext(0.5, 0.01, data['caption'].iloc[3], wrap=True, horizontalalignment='center', fontsize=12)


plt.show()