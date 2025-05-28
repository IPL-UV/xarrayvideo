from huggingface_hub import HfApi, login
from pathlib import Path
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

# Load the environment variables
load_dotenv()
api = HfApi()

def upload_taco(taco_path: Path, taco_name: str, repo_id: str):
    # Ensure taco_path is the directory containing the .taco file(s)
    # and taco_name is the prefix of the .taco file(s)
    
    # If taco_path is a file, use its parent directory and its name as taco_name
    if taco_path.is_file() and taco_path.name.startswith(taco_name) and taco_path.suffix == '.taco':
        files_to_upload = [str(taco_path)]
        taco_dir = taco_path.parent
    # If taco_path is a directory, glob for files starting with taco_name
    elif taco_path.is_dir():
        files_to_upload = [str(file) for file in taco_path.glob(f"{taco_name}*.taco")]
        taco_dir = taco_path
    else:
        print(f"Error: taco_path '{taco_path}' is neither a valid .taco file nor a directory, or taco_name '{taco_name}' does not match.")
        return

    if not files_to_upload:
        print(f"No .taco files found in '{taco_dir}' with prefix '{taco_name}'.")
        return

    for i,file_path_str in enumerate(tqdm(files_to_upload)):
        api.upload_file(
            path_or_fileobj=file_path_str,
            repo_id=repo_id,
            path_in_repo=file_path.name,
            repo_type="dataset",
            commit_message=f"Upload {file_path.name}"
        )

        print(f"File uploaded: {file_path.name} to {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload TACO datasets to Hugging Face Hub.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["deepextremecubes", "dynamicearthnet", "simples2"],
        required=True,
        help="Specify which dataset to upload: 'deepextremecubes' or 'dynamicearthnet' or 'simples2' (though not a taco)"
    )
    args = parser.parse_args()

    if args.dataset == "deepextremecubes":
        # This taco_path should be the specific .taco file or the directory containing it.
        taco_path = Path("/scratch/users/databases") 
        taco_name = "DeepExtremeCubes-video" # This will be used if taco_path is a directory
        repo_id = "isp-uv-es/DeepExtremeCubes-video"
        upload_taco(taco_path, taco_name, repo_id)

    elif args.dataset == "dynamicearthnet":
        # This taco_path should be the specific .taco file.
        taco_path = Path("/scratch/users/databases/dynamicearthnet-video-final/DynamicEarthNet-video.taco")
        repo_id = "isp-uv-es/DynamicEarthNet-video"
        upload_taco(taco_path, repo_id)

    elif args.dataset == 'simples2':
        files_path = Path("/home/oscar/cubos_julio")
        repo_id = "isp-uv-es/SimpleS2"
        for i,file_path in enumerate(tqdm(files_path.iterdir())):
            if file_path.is_dir() or file_path.name.startswith("."): continue
            api.upload_file(
                path_or_fileobj=file_path,
                repo_id=repo_id,
                path_in_repo=file_path.name,
                repo_type="dataset",
                commit_message=f"Upload {file_path.name}"
            )