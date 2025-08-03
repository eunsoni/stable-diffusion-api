import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from transformers import CLIPTokenizer

app = FastAPI()

# 저장된 경로
load_dir = "./saved_sd15"

# 구성요소 불러오기
unet = torch.load(os.path.join(load_dir, "unet.pth"), map_location="cuda").eval()
vae = torch.load(os.path.join(load_dir, "vae.pth"), map_location="cuda").eval()
text_encoder = torch.load(os.path.join(load_dir, "text_encoder.pth"), map_location="cuda").eval()
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(load_dir, "tokenizer"))

# Scheduler 정의
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    steps_offset=1,
    clip_sample=False
)

# 파이프라인 조립
pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
).to("cuda")


# 이미지 생성 함수
def generate_image(prompt: str, init_image: Image.Image = None) -> io.BytesIO:
    if init_image is None:
        raise ValueError("init_image is required for img2img pipeline.")
    
    with torch.inference_mode():
        init_image = init_image.resize((512, 512))
        result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)
        image = result.images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# API 엔드포인트
@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):
    if image is None:
        raise ValueError("Image input is required for img2img.")

    contents = await image.read()
    init_image = Image.open(io.BytesIO(contents)).convert("RGB")

    buffer = generate_image(prompt, init_image)
    return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 