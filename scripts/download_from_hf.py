from huggingface_hub import snapshot_download

repo_id = "isp-uv-es/DeepExtremeCubes-video"

local_dir = "/data/databases/DeepExtremeCubes-video"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # safer for datasets
    resume_download=True,
)

print("Done")