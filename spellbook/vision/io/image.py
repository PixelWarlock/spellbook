import numpy as np
from PIL import Image
from typing import Union, Any

def load_image(filepath:str, return_as_array:bool=True)->Any[Image, np.array]:
    image = Image.open(filepath).convert('RGB')
    if return_as_array:
        return np.array(filepath)

def save_image(filepath:str, image:Union[np.array, Image])->None:
    if isinstance(image,Image):
        Image.save(filepath)
    elif isinstance(image, np.array):
        Image.fromarray(image).save(filepath)

