import os
import cv2
import PIL.Image as Image
import numpy as np


def blur_padding(img, kernel=3, iteration=32, mask=None, save_path=None):
    # get RGB and mask
    image = img[..., :3] / 255
    
    if mask is None:
        # mask = img[..., 3, np.newaxis] / 255 > 0.
        mask = (img[..., :3] / 255).mean(-1)
        mask[mask > 0] = 1.0
        mask = mask[..., np.newaxis]
        #print(mask.shape)
        
    # masking
    image   = image * mask
    new_img = image.copy()
    new_msk = mask.copy()
    
    for _ in range(iteration):
        # kernel
        dilate_k = np.ones((kernel,kernel))
        
        # each color
        dilate_r = cv2.dilate(new_img[..., 0, np.newaxis], dilate_k, iterations=1)  # make dilation image
        dilate_g = cv2.dilate(new_img[..., 1, np.newaxis], dilate_k, iterations=1)  # make dilation image
        dilate_b = cv2.dilate(new_img[..., 2, np.newaxis], dilate_k, iterations=1)  # make dilation image
        
        # mask
        dilate_m = cv2.dilate(new_msk, dilate_k, iterations=1)  # make dilation image        
        dilate_m = dilate_m[...,np.newaxis]
        dilate_m = dilate_m - new_msk
        
        # concatenate all channel
        dilate_image = np.concatenate((
                dilate_r[...,np.newaxis], 
                dilate_g[...,np.newaxis], 
                dilate_b[...,np.newaxis]
            ),axis=2)
        
        # mask for only dilated region
        dilate_image = dilate_image * dilate_m
        
        new_img = new_img + dilate_image
        new_msk = new_msk + dilate_m
        
    new_img = cv2.GaussianBlur(new_img, (7, 7), 0)
    new_img = new_img * (1-mask) + image
    
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        Image.fromarray(new_img).save(f'{save_path}/blurred.png')
        
    return new_img
        
if __name__ == '__main__':
    img = np.array(Image.open(r"D:\test\py-opengl\tmp\ICT_Backhead.png"))
    _ = blur_padding(img, iteration=64, save_path='./tmp')