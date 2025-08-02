from huggingface_hub import snapshot_download

print("📥 FLUX 모델 다운로드 중...")
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="./models/FLUX",
    local_dir_use_symlinks=False
)

print("📥 Ghibli LoRA 다운로드 중...")
snapshot_download(
    repo_id="openfree/flux-chatgpt-ghibli-lora",
    local_dir="./models/ghibli_lora",
    local_dir_use_symlinks=False
)

print("✅ 모델 다운로드 완료")