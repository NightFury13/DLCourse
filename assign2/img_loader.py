from PIL import Image
import numpy as np

class ImageLoader(object):
    def __init__(self):
        self.version = '0.1'

    def get_img_numpy(self, img_paths):
        return np.array([np.array(Image.open(img_path)) for img_path in img_paths])
