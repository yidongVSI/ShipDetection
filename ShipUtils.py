# Utility functions
import matplotlib.pyplot as plt
from skimage.data import imread
import numpy as np
from ShipDataset import rle_encode, rle_decode

def PlotShipImages(imgs = [], masks = [], size = 20):
  '''
  Visualize training image along with their annotation
  Args:
    imgs:  list of numpy array with size (W x H x 3) -- An RGB image
    masks: list of numpy array with size (W x H) -- associated annotation
    size:  int, size of each individual plot
  '''
  if len(imgs) != len(masks):
    print("WARNING: PlotSingleImage -- input size mismatch!!!, plot nothing")
    return
  c = len(imgs)
  f, a = plt.subplots(2, c, figsize = (size, 5))
  for i in range(c):
    a[0][i].imshow(imgs[i])
    a[1][i].imshow(masks[i], cmap = 'gray')
  return


def PlotShipMask(ImageId, masks):
    '''
    Visualize ship with mask.
    Args:
        ImageId: string, Image id.
        masks: dataframe with two columns, ImageId and EncodedPixels
    '''
    img = imread('../input/train/' + ImageId)
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    all_masks = np.zeros((768, 768))    
    if type(img_masks[0]) is str:
        # Take the individual ship masks and create a single mask array for all ships
        for mask in img_masks:
            all_masks += rle_decode(mask, shape=(768, 768))

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
    return