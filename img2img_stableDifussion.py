# pip install https://github.com/huggingface/diffusers/archive/main.zip -qUU --force-reinstall
# pip install -qq -U transformers ftfy
# pip install -qq "ipywidgets>=7,<8"
# pip install Image
# pip install --ignore-installed Pillow==9.0.0
# pip install accelerate

import os
from torch import autocast
import torch
import requests
from PIL import Image
import uuid
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token="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"
).to(device)


def img2img(img_path, prompt):
    imgobject = Image.open(img_path)
    img_path_lst = []
    abs_path = os.path.dirname(os.path.realpath(__file__)) + "/images/"
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
    init_image = imgobject.convert("RGB")
    init_image = init_image.resize((768, 512))
    with autocast("cuda"):
        images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images
        file_name = str(uuid.uuid4())
        img_path = abs_path + f"{file_name}.jpg"
        images.save(img_path)
        img_path_lst.append(img_path)
    return img_path


if __name__ == "__main__":
    # imgobject = Image.open("test.jpg")
    img_path = img2img("test.jpg", "A very beautiful anime girl, full body, long wavy blond hair, sky blue eyes")
    print(img_path)