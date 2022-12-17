import os
import argparse
import datetime
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=200,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=5,
    help='how many samples to produce for each given prompt',
)
opt = parser.parse_args()

#prompt = "anime of tsundere moe kawaii beautiful girl"
prompt = 'anime of tsundere moe kawaii beautiful girl'
negative_prompt = None
#negative_prompt = "red eyes red hair"
#num_images_per_prompt = 5

model_id = "./stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

generator = torch.Generator(device="cuda").manual_seed(opt.seed)
images = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    generator = generator,
    num_images_per_prompt = opt.n_samples).images

os.makedirs('results', exist_ok=True)

now = datetime.datetime.today()
now_str = now.strftime('%m%d_%H%M')
for i, image in enumerate(images):
    image.save(os.path.join('results', f'{now_str}_{i}.png'))


