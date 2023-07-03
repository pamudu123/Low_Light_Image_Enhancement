''' 
Date: May 13, 2023

Description :
This script is used to add padding to the image before passing it to 
the neural network and remove the padding from the output image generated 
by the neural network

Method 1 - Center padding
This method adds padding to the edges of the image such that the image remains centered after padding.

Method 2 - Corner Padding:
This method adds padding to the right and the bottom of the image

set the 'padding_method' variable to either 'center' or 'corner' before running the script
'''

import numpy as np
import math

def padding_calc(input_dim,multiplier=16):
  return math.ceil(input_dim/multiplier)*multiplier - input_dim

# Add Padding
def pad_image(image,mood = "center_padding"):
  img_h = image.shape[0]
  img_w = image.shape[1]
  
  pad_y = padding_calc(img_h)
  pad_x = padding_calc(img_w)
  
  if mood == "center_padding":
    pad_y2 = pad_y//2
    pad_x2 = pad_x//2

    padded_img = image.copy()
    if pad_y%2 != 0:
      padded_img = np.pad(image, ((pad_y2, pad_y2+1), (0, 0), (0, 0)), mode='constant')
    if pad_x%2 != 0:
      padded_img = np.pad(image, ((0, 0), (pad_x2, pad_x2+1), (0, 0)), mode='constant')
    if pad_y%2 == 0 & pad_x%2 == 0:
      padded_img = np.pad(image, ((pad_y2, pad_y2), (pad_x2, pad_x2), (0, 0)), mode='constant')

  elif mood == "corner_padding":
    padded_img = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')

  return padded_img

# Remove Padding
def inverse_padding(pad_image,image_dim,pad_method="center_padding"):
  pad_img_height = pad_image.shape[0]
  pad_img_width = pad_image.shape[1]
  
  img_height = image_dim[0]
  img_width = image_dim[1]
  
  if pad_method == "center_padding":
    pad_y1 = (pad_img_height - img_height)//2
    if pad_y1*2 == (pad_img_height - img_height):pad_y2 = pad_y1
    else: pad_y2 = pad_y1+1

    pad_x1 = (pad_img_width - img_width)//2
    if pad_x1*2 == (pad_img_width - img_width):pad_x2 = pad_x1
    else: pad_x2 = pad_x1+1
    extract_image = pad_image[pad_y1:pad_img_height-pad_y2,pad_x1:pad_img_width-pad_x2]
  
  if pad_method == "corner_padding":
    extract_image = pad_image[0:img_height,0:img_width]

 
  return extract_image



