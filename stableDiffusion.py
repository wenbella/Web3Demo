# !pip install diffusers==0.3.0
# !pip install transformers scipy ftfy
# !pip install "ipywidgets>=7,<8"



import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import ipfshttpclient
from PIL import Image
from helper_script import upload_img_to_ipfs
from collections import defaultdict

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16",
                                               torch_dtype=torch.float16, use_auth_token=True)
pipe = pipe.to("cuda")


def outputImage(text):
    client = ipfshttpclient.connect()
    abs_path = os.path.dirname(os.path.realpath(__file__)) + "/images/"
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
    ret = defaultdict()
    for i in range(0, 4):
        prompt = text
        with autocast("cuda"):
            image = pipe(prompt).images[0]
        img_path = abs_path + f"output%s.png" % str(i)
        if os.path.exists(img_path):
            os.remove(img_path)
        image.save(img_path)
        ret = client.add(img_path)
        ipfs_hash = ret['Hash']
        image_url = f"https://ipfs.io/ipfs/{ipfs_hash}"
        ret[f'image%s' % str(i)] = image_url
    return ret


if __name__ == "__main__":
    prompt = "2. detailed digital art by Disney, Pixar, beautiful celestial woman, celestial, pastel lavender hair in loose curls, bioluminescent purple clothes made of stars, Disney Pixar eyes, deep look, anime, in the style of Kazushi Hagiwara, Alphonse Mucha, Theodor Suess Geisel, Ilya Kuvshinov, Lisa Frank, Gil Elvgren"
    print(outputImage(prompt))
