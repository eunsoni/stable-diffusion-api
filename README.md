# Stable Diffusion Image Generation API

FastAPI와 Stable Diffusion을 사용한 고성능 이미지 변환 API입니다. 기존 이미지와 텍스트 프롬프트를 통해 새로운 스타일의 이미지를 생성할 수 있는 RESTful API를 제공합니다.

## � 목차

- [🚀 주요 기능](#-주요-기능)
- [📋 시스템 요구사항](#-시스템-요구사항)
- [🛠️ 설치 및 실행](#️-설치-및-실행)
- [☁️ AWS 인스턴스 설정 가이드](#️-aws-인스턴스-설정-가이드)
- [📚 API 사용법](#-api-사용법)
- [📖 API 문서](#-api-문서)
- [⚙️ 설정 및 환경 변수](#️-설정-및-환경-변수)
- [🔧 개발 및 기여](#-개발-및-기여)
- [🛠️ 문제 해결](#️-문제-해결)

## �🚀 주요 기능

- **Image-to-Image**: 기존 이미지를 기반으로 한 이미지 변환 (img2img)
- **텍스트 프롬프트**: 상세한 텍스트 설명을 통한 이미지 스타일 변경
- **RESTful API**: 간단하고 직관적인 API 인터페이스
- **GPU 가속**: NVIDIA GPU를 활용한 빠른 이미지 생성
- **Docker 지원**: 컨테이너화된 배포로 쉬운 설치 및 관리
- **자동 문서화**: Swagger UI 및 ReDoc을 통한 API 문서

## 📋 시스템 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (최소 8GB VRAM 권장)
- **RAM**: 최소 16GB (32GB 권장)
- **저장공간**: 최소 10GB (모델 파일 포함)

### 소프트웨어
- **Docker**: 20.10 이상
- **Docker Compose**: 1.29 이상
- **NVIDIA Container Toolkit**: GPU 지원을 위해 필요
- **CUDA**: 11.8 이상 (Docker 이미지에 포함)

### 지원 OS
- Ubuntu 18.04 이상
- CentOS 7 이상
- Windows 10/11 (WSL2 + Docker Desktop)

## 🛠️ 설치 및 실행

### 사전 준비사항

1. **NVIDIA Container Toolkit 설치** (GPU 지원을 위해 필요)
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **GPU 확인**
```bash
nvidia-smi
```

### 🐳 Docker를 이용한 실행 (권장)

#### 1. 저장소 클론
```bash
git clone https://github.com/eunsoni/stable-diffusion-api.git
cd stable-diffusion-api
```

#### 2. 모델 파일 준비
이 API는 사전 훈련된 Stable Diffusion 1.5 모델을 사용합니다. 
모델 파일들이 `/mnt/efs/saved_sd15` 경로에 있어야 합니다.

```bash
# 모델 다운로드 (선택사항 - 자동으로 다운로드됨)
python download_models.py
```

#### 3. Docker Compose로 실행
```bash
# 백그라운드에서 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

## ☁️ AWS 인스턴스 설정 가이드 (선택사항)

새로운 AWS 인스턴스에서 EFS를 사용하여 이 프로젝트를 실행하는 전체 설정 과정입니다.

### 1. Docker 설치

```bash
# Docker 설치를 위한 필수 패키지를 설치합니다
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker의 공식 GPG 키를 추가합니다
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker의 공식 저장소를 추가합니다
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 엔진과 관련 패키지를 설치합니다
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker가 제대로 설치되었는지 확인합니다
sudo systemctl status docker

# sudo 없이 Docker 명령어를 실행하려면 현재 사용자를 docker 그룹에 추가합니다
sudo usermod -aG docker $USER
newgrp docker
```

### 2. NVIDIA 드라이버 설치 (GPU 인스턴스의 경우)

```bash
# NVIDIA 드라이버 자동 설치
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# 재접속 후 드라이버 확인
nvidia-smi
```

### 3. NVIDIA Container Toolkit 설치

```bash
# NVIDIA Container Toolkit 저장소 추가
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# NVIDIA Container Toolkit 설치
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# GPU 인식 테스트 (선택사항)
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### 4. EFS 설치 및 마운트

```bash
# NFS 클라이언트 설치
sudo apt update
sudo apt install -y nfs-common

# EFS 마운트 디렉토리 생성
sudo mkdir -p /mnt/efs

# EFS 마운트 (실제 EFS ID 예시)
sudo mount -t nfs4 -o nfsvers=4.1 fs-040df8c9ef7b96af3.efs.ap-northeast-2.amazonaws.com:/ /mnt/efs

# 정상 마운트 확인
df -h
```

### 5. EFS 자동 마운트 설정 (선택사항)

인스턴스 재부팅 시 EFS를 자동으로 마운트하려면:

```bash
# fstab 파일 편집
sudo nano /etc/fstab

# 아래 라인 추가 (실제 EFS ID로 변경)
fs-040df8c9ef7b96af3.efs.ap-northeast-2.amazonaws.com:/ /mnt/efs nfs4 defaults,_netdev 0 0
```

### 6. 사전 빌드된 Docker 이미지 실행 (권장)

가장 간단한 방법은 사전 빌드된 Docker 이미지를 사용하는 것입니다:

```bash
# Docker 이미지 실행 (자동으로 Docker Hub에서 다운로드)
docker run --rm -it --gpus all \
  -v /mnt/efs/saved_sd15:/mnt/efs/saved_sd15 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8000:8000 \
  eunsunhub/stable-diffusion-api:efs

# 백그라운드 실행
docker run -d --name stable-diffusion-api --gpus all \
  -v /mnt/efs/saved_sd15:/mnt/efs/saved_sd15 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8000:8000 \
  --restart unless-stopped \
  eunsunhub/stable-diffusion-api:efs

# 컨테이너 상태 확인
docker ps

# 로그 확인
docker logs -f stable-diffusion-api
```

### 7. 소스코드에서 직접 빌드 (선택사항)

소스코드를 수정하거나 커스터마이징이 필요한 경우:

```bash
# 프로젝트 클론
git clone https://github.com/eunsoni/stable-diffusion-api.git
cd stable-diffusion-api

# Docker Compose로 실행
docker-compose up -d --build

# 서비스 상태 확인
docker-compose ps
```

### AWS 보안 그룹 설정

API 서버에 외부에서 접근하려면 보안 그룹에서 포트 8000을 허용해야 합니다:

- **포트**: 8000
- **프로토콜**: TCP
- **소스**: 0.0.0.0/0 (모든 IP 허용) 또는 특정 IP 범위

### 🐍 로컬 환경에서 실행

#### 1. Python 가상환경 생성
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

#### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

#### 3. 서버 실행
```bash
python main.py
```

서버가 정상적으로 시작되면 `http://localhost:8000`에서 API에 접근할 수 있습니다.

## 📚 API 사용법

### 기본 엔드포인트
- **Base URL**: `http://localhost:8000`
- **이미지 변환**: `POST /generate`

### 이미지 변환 API

이 API는 **Image-to-Image (img2img)** 기능만 지원합니다. 기존 이미지를 업로드하고 텍스트 프롬프트를 제공하면 새로운 스타일의 이미지로 변환됩니다.

#### 요청 파라미터
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `prompt` | string | 필수 | 이미지 생성을 위한 텍스트 프롬프트 |
| `image` | file | 필수 | 변환할 기본 이미지 (img2img) |

#### cURL 예제

**기본 이미지 변환**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=a beautiful sunset over mountains, oil painting style" \
  -F "image=@input_image.jpg" \
  --output generated_image.png
```

**복잡한 프롬프트 예제**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=professional portrait photo of a cyberpunk character, neon lights, detailed face, 4k, high quality" \
  -F "image=@portrait.jpg" \
  --output cyberpunk_portrait.png
```

#### Python 클라이언트 예제

```python
import requests

# 기본 사용법
def generate_image(prompt, image_path):
    url = "http://localhost:8000/generate"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'prompt': prompt}
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            with open('generated_image.png', 'wb') as output:
                output.write(response.content)
            print("이미지가 성공적으로 생성되었습니다!")
        else:
            print(f"오류 발생: {response.status_code}")

# 사용 예제
generate_image(
    prompt="a serene landscape with mountains and lake, impressionist style",
    image_path="input.jpg"
)
```

#### JavaScript/Node.js 예제

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function generateImage(prompt, imagePath) {
    const form = new FormData();
    form.append('prompt', prompt);
    form.append('image', fs.createReadStream(imagePath));
    
    try {
        const response = await axios.post('http://localhost:8000/generate', form, {
            headers: form.getHeaders(),
            responseType: 'stream'
        });
        
        response.data.pipe(fs.createWriteStream('generated_image.png'));
        console.log('이미지가 성공적으로 생성되었습니다!');
    } catch (error) {
        console.error('오류 발생:', error.message);
    }
}

// 사용 예제
generateImage(
    "a futuristic city skyline at night, cyberpunk style",
    "input.jpg"
);
```

### 응답 형식

**성공 응답 (200 OK)**
- Content-Type: `image/png`
- Body: 생성된 이미지 바이너리 데이터

**오류 응답**
```json
{
    "detail": "오류 메시지"
}
```

### 프롬프트 작성 팁

#### 효과적인 프롬프트 구조 (img2img용)
```
[원하는 스타일] + [품질 키워드] + [분위기/조명] + [기술적 세부사항]
```

#### 이미지 변환 예제
- `"oil painting style, highly detailed, warm lighting, masterpiece"` - 유화 스타일로 변환
- `"watercolor painting, soft colors, artistic, high quality"` - 수채화 스타일로 변환
- `"cyberpunk style, neon lights, futuristic, detailed, 4k"` - 사이버펑크 스타일로 변환
- `"pencil sketch, black and white, artistic drawing, detailed"` - 연필 스케치로 변환
- `"anime style, colorful, detailed face, high quality"` - 애니메이션 스타일로 변환

#### 품질 향상 키워드
- `highly detailed`, `4k`, `8k`, `masterpiece`, `high quality`
- `professional photography`, `studio lighting`, `dramatic lighting`
- `artstation`, `concept art`, `digital art`, `fine art`

## 📖 API 문서

서버 실행 후 다음 URL에서 대화형 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
  - 대화형 API 테스트 인터페이스
  - 실시간으로 API 호출 테스트 가능
  
- **ReDoc**: http://localhost:8000/redoc
  - 깔끔한 API 문서 뷰
  - 상세한 스키마 정보 제공

## ⚙️ 설정 및 환경 변수

### 환경 변수

| 변수명 | 기본값 | 설명 |
|-------|--------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 디바이스 번호 |
| `MODEL_PATH` | `/mnt/efs/saved_sd15` | 모델 파일 저장 경로 |
| `HOST` | `0.0.0.0` | 서버 호스트 주소 |
| `PORT` | `8000` | 서버 포트 번호 |

### Docker Compose 설정 사용자 정의

`docker-compose.yml` 파일을 수정하여 설정을 변경할 수 있습니다:

```yaml
version: '3.8'

services:
  stable-diffusion-api:
    build: .
    ports:
      - "8000:8000"  # 포트 변경 시 수정
    environment:
      - CUDA_VISIBLE_DEVICES=0  # GPU 설정
    runtime: nvidia
    volumes:
      - /your/model/path:/mnt/efs/saved_sd15  # 모델 경로 변경
    restart: unless-stopped
```

## 🔧 개발 및 기여

### 개발 환경 설정

1. **개발 의존성 설치**
```bash
pip install -r requirements.txt
pip install pytest black flake8
```

2. **코드 포맷팅**
```bash
black .
flake8 .
```

3. **테스트 실행**
```bash
pytest
```

### 프로젝트 구조

```
stable-diffusion-api/
├── main.py                 # FastAPI 애플리케이션 메인 파일
├── download_models.py      # 모델 다운로드 스크립트
├── requirements.txt        # Python 의존성
├── Dockerfile             # Docker 이미지 빌드 설정
├── docker-compose.yml     # Docker Compose 설정
└── README.md              # 프로젝트 문서
```

### 주요 의존성

| 패키지 | 버전 | 용도 |
|-------|------|------|
| `fastapi` | 0.104.1 | 웹 API 프레임워크 |
| `torch` | 2.2.2 | 딥러닝 프레임워크 |
| `diffusers` | 0.34.0 | Stable Diffusion 파이프라인 |
| `transformers` | 4.54.1 | 트랜스포머 모델 |
| `Pillow` | 10.1.0 | 이미지 처리 |

## 🚀 성능 최적화

### GPU 메모리 최적화
```python
# 메모리 효율적인 설정
pipe.enable_memory_efficient_attention()
pipe.enable_xformers_memory_efficient_attention()
```

### 배치 처리
여러 이미지를 동시에 생성할 때는 배치 처리를 고려하세요:

```python
# 여러 프롬프트 동시 처리
prompts = ["prompt1", "prompt2", "prompt3"]
images = pipe(prompts, num_inference_steps=50)
```

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족
```
CUDA out of memory
```
**해결방법:**
- 이미지 해상도 낮추기 (512x512 권장)
- 배치 크기 줄이기
- `torch.cuda.empty_cache()` 사용

#### 2. 모델 로딩 실패
```
FileNotFoundError: Model files not found
```
**해결방법:**
- 모델 경로 확인: `/mnt/efs/saved_sd15`
- `download_models.py` 실행하여 모델 다운로드

#### 3. Docker 권한 문제
```
Permission denied
```
**해결방법:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### 4. NVIDIA Runtime 오류
```
could not select device driver with capabilities: [[gpu]]
```
**해결방법:**
- NVIDIA Container Toolkit 재설치
- Docker 재시작: `sudo systemctl restart docker`

### 로그 확인

```bash
# Docker 로그 확인
docker-compose logs -f

# 특정 컨테이너 로그
docker logs <container_id>
```

## 📊 모니터링

### GPU 사용량 모니터링
```bash
# 실시간 GPU 상태 확인
watch -n 1 nvidia-smi

# GPU 메모리 사용량 확인
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### API 성능 모니터링
- 응답 시간 측정
- 동시 요청 처리 능력 테스트
- 메모리 사용량 모니터링

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시다면:

1. 이 저장소를 Fork하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push하세요 (`git push origin feature/AmazingFeature`)
5. Pull Request를 열어주세요

### 기여 가이드라인
- 코드 스타일: Black 포맷터 사용
- 테스트: 새로운 기능에 대한 테스트 추가
- 문서화: README 및 코드 주석 업데이트

## 📞 지원 및 연락

- **이슈 리포트**: [GitHub Issues](https://github.com/eunsoni/stable-diffusion-api/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/eunsoni/stable-diffusion-api/discussions)
- **이메일**: your-email@example.com

## 🔗 관련 링크

- [Stable Diffusion 공식 문서](https://huggingface.co/docs/diffusers/index)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Docker 설치 가이드](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [Stability AI](https://stability.ai/) - Stable Diffusion 모델 제공
- [Hugging Face](https://huggingface.co/) - Diffusers 라이브러리 및 모델 호스팅
- [FastAPI](https://fastapi.tiangolo.com/) - 훌륭한 웹 프레임워크

---

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!** 
