# !pip install diffusers==0.3.0
# !pip install transformers scipy ftfy
# !pip install "ipywidgets>=7,<8"



import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

pipe = pipe.to("cuda")

from PIL import Image
from torch import autocast


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def outputImage(text):
  num_cols = 3
  num_rows = 4

  prompt = [text] * num_cols

  all_images = []
  for i in range(num_rows):
    with autocast("cuda"):
      images = pipe(prompt).images
    all_images.extend(images)

  grid = image_grid(all_images, rows=num_rows, cols=num_cols)
  grid.save(f"output.png")