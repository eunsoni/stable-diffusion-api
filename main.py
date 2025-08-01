from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import io

app = FastAPI()

# 로컬 경로에 다운로드한 모델을 로드
MODEL_DIR = "./models/SD15"  # 필요에 따라 경로 수정

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16
).to("cuda")

# 예: UNet 바이어스 수정 (자유롭게 커스터마이즈 가능)
with torch.no_grad():
    pipe.unet.conv_in.bias += 0.01


# 이미지 생성 함수 (img2img)
def generate_image(prompt: str, init_image: Image.Image = None) -> io.BytesIO:
    if init_image:
        init_image = init_image.resize((512, 512))
        image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    else:
        image = pipe(prompt).images[0]

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
    init_image = None
    if image:
        contents = await image.read()
        init_image = Image.open(io.BytesIO(contents)).convert("RGB")

    buffer = generate_image(prompt, init_image)
    return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 