from lib2to3.pytree import NegatedPattern
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "./stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

#prompt = "anime of tsundere moe kawaii beautiful girl"
prompt = 'Hatsune Miku in anime style cooking in the beach,high quality,pixiv'
negative_prompt = None
#negative_prompt = "red eyes red hair"
seed = 200
num_images_per_prompt = 5

generator = torch.Generator(device="cuda").manual_seed(seed)
images = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    generator = generator,
    num_images_per_prompt = num_images_per_prompt).images

for i, image in enumerate(images):
    image.save(f"result_{i}.png")
