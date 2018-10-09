# Basid pytorth based Dataset

from __future__ import print_function, division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import skimage.measure as skm

# Run-Length Encode and Decode
# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape, pos_val = 255):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, pos_val - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = pos_val
    return img.reshape(shape).T


class ShipDataset(Dataset):
  '''A pytroch based Dataset for Kaggle Airbus Ship Detection challenge
  '''
  def __init__(self, csv_file, img_dir, transform = None):
    '''Args:
      csv_file (string): Path to the csv annotation (RLE encoded by default)
      img_dir  (string): Path to the imagery (train / test)
      transform (callable optional):  Optional transform that can be applied on a sample
    '''
    assert os.path.isfile(csv_file), "invalid input csv_file: {}".format(csv_file)
    assert os.path.exists(img_dir),  "invalid input img_dir: {}".format(img_dir)
    super(ShipDataset, self).__init__()
    self.img_dir = img_dir
    self.csv = csv_file
    self.df = pd.read_csv(csv_file)
    self.transform = transform
    self.img_files = glob.glob(img_dir + "/*.jpg")
    self.img_names = []
    for img_file in self.img_files:
      self.img_names.append(os.path.basename(img_file))

  def __len__(self):
    return len(self.img_files)

  def __getitem__(self, idx):
    '''Args:
        idx (int): Index
       Returns:
        Two numpy arries: first is the RGB image (H x W x 3) and second is (H x W x 1) the annotation mask image
    '''
    if idx > len(self.img_files):
        return None, None
    img_file = self.img_files[idx]
    img = np.asarray(Image.open(img_file))
    ser = self.df.loc[self.df['ImageId'] == os.path.basename(img_file)]
    encode_all = ser.EncodedPixels.values
    mask_all = []
    for encode in encode_all:
      if pd.isna(encode):
        mask = np.zeros( (img.shape[0:2]), dtype=img.dtype)
      else:
        mask = rle_decode(encode, img.shape[0:2])
      mask_all.append(mask)
    # add all labels
    sum_mask = mask_all[0]
    for i in range(1, len(mask_all)):
        sum_mask = np.add(sum_mask, mask_all[i])
    sum_mask[sum_mask > 0] = 255
    if self.transform:
      img = self.transform(img)
    return img, sum_mask

  def get_image(self, img_name):
    '''Args:
      img_name (str): some image name
      Returns:
        Specified image along with its label mask
    '''
    img_file = os.path.join(self.img_dir, img_name)
    if not os.path.isfile(img_file):
      return None, None
    img = np.asarray(Image.open(img_file))
    ser = self.df.loc[self.df['ImageId'] == os.path.basename(img_file)]
    encode_all = ser.EncodedPixels.values
    mask_all = []
    for encode in encode_all:
      if pd.isna(encode):
        mask = np.zeros( (img.shape[0:2]), dtype=img.dtype)
      else:
        mask = rle_decode(encode, img.shape[0:2])
      mask_all.append(mask)
    # add all labels
    sum_mask = mask_all[0]
    for i in range(1, len(mask_all)):
        sum_mask = np.add(sum_mask, mask_all[i])
    sum_mask[sum_mask > 0] = 255
    if self.transform:
      img = self.transform(img)
    return img, sum_mask

  def get_footprint(self, img_name):
    '''Args:
      img_name (str): some image name
      Returns:
        Series of bounding box around each segmented ship
    '''
    img_file = os.path.join(self.img_dir, img_name)
    if not os.path.isfile(img_file):
      return None, None
    img = np.asarray(Image.open(img_file))
    ser = self.df.loc[self.df['ImageId'] == os.path.basename(img_file)]
    encode_all = ser.EncodedPixels.values
    footprint_all = []
    for encode in encode_all:
      # gen binary mask
      if pd.isna(encode):
        continue
      mask = rle_decode(encode, img.shape[0:2])
      label_img = skm.label(mask)
      for region in skm.regionprops(label_img):
        if region.area > 0:
          minr, minc, maxr, maxc = region.bbox

          footprint_all.append((minc, minr, maxc, maxr))
    return img, footprint_all

