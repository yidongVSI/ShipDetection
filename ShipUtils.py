# Utility functions
import matplotlib.pyplot as plt

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