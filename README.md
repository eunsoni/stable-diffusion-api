# Stable Diffusion Image Generation API

FastAPI와 Stable Diffusion을 사용한 이미지 생성 API입니다.

## 기능

- 텍스트 프롬프트를 통한 이미지 생성
- 기존 이미지를 기반으로 한 이미지 변환 (img2img)
- RESTful API 인터페이스

## 요구사항

- Docker와 Docker Compose
- NVIDIA GPU (CUDA 지원)
- NVIDIA Container Toolkit

## 설치 및 실행

### 1. 저장소 클론
```bash
git clone <your-repository-url>
cd stable-diffusion-api
```

### 2. Docker Compose로 실행
```bash
docker-compose up --build
```

### 3. 로컬 실행 (선택사항)
```bash
pip install -r requirements.txt
python main.py
```

## API 사용법

### 이미지 생성
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=a beautiful sunset over mountains"
```

### 이미지 변환 (img2img)
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=a beautiful sunset over mountains" \
  -F "image=@input_image.jpg"
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 환경 변수

- `CUDA_VISIBLE_DEVICES`: 사용할 GPU 설정 (기본값: 0)

## 라이선스

MIT License 