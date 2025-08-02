# CUDA 지원 PyTorch 베이스 이미지 사용
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY main.py .
COPY download_models.py .

# 모델 다운로드 (빌드 시 1회)
RUN python download_models.py

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]