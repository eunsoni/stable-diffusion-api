from diffusers import StableDiffusionPipeline
import torch
import os

# 저장 경로
save_dir = "./saved_sd15"
os.makedirs(save_dir, exist_ok=True)

# Hugging Face에서 모델 로드
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

# 구성요소 저장
torch.save(pipe.unet, os.path.join(save_dir, "unet.pth"))
torch.save(pipe.vae, os.path.join(save_dir, "vae.pth"))
torch.save(pipe.text_encoder, os.path.join(save_dir, "text_encoder.pth"))

# tokenizer는 Hugging Face 방식으로 저장
pipe.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

print("저장 완료")