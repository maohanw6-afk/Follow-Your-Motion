
import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from diffusers.utils import export_to_video      
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)



flow_shift = 5.0 
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)

pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")           
pipe.vae.enable_tiling()  
           

prompt = "A glass of milk on a dining table captured with a zoom in"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=720,         
     width=1280, 
     num_frames=81,           
     num_inference_steps=35, 
     guidance_scale=7.0,  
    ).frames[0]


export_to_video(output, "result_video.mp4", fps=16)
