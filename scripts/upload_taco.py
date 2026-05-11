from huggingface_hub import HfApi
from pathlib import Path
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

# Load the environment variables
load_dotenv()
api = HfApi()

def remove_existing_files(repo_id: str, file_extension: str = "taco"):
    """Delete files from the HF dataset repo filtered by suffix, or all files."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if file_extension == "*":
            files_to_delete = files
            description = "all files"
        else:
            suffix = f".{file_extension.lstrip('.')}"
            files_to_delete = [f for f in files if f.endswith(suffix)]
            description = f"'{suffix}' files"

        if not files_to_delete:
            print(f"No existing {description} found in {repo_id}.")
            return

        print(f"Removing {len(files_to_delete)} existing {description} from {repo_id}...")
        for fname in files_to_delete:
            api.delete_file(path_in_repo=fname, repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Remove {fname}")
            print(f"  Deleted: {fname}")
    except Exception as e:
        print(f"Error removing files from {repo_id}: {e}")

def upload_taco(taco_path: Path, taco_name: str, repo_id: str, file_extension: str = "tacozip"):
    suffix = f".{file_extension.lstrip('.')}"

    # Ensure taco_path is the directory containing the target file(s)
    # and taco_name is the prefix of the target file(s)
    
    # If taco_path is a file, use its parent directory and its name as taco_name
    if taco_path.is_file() and taco_path.name.startswith(taco_name) and taco_path.suffix == suffix:
        files_to_upload = [taco_path]
        taco_dir = taco_path.parent
    # If taco_path is a directory, glob for files starting with taco_name
    elif taco_path.is_dir():
        files_to_upload = [file for file in taco_path.glob(f"{taco_name}*{suffix}")]
        taco_dir = taco_path
    else:
        print(f"Error: taco_path '{taco_path}' is neither a valid {suffix} file nor a directory, or taco_name '{taco_name}' does not match.")
        return

    if not files_to_upload:
        print(f"No {suffix} files found in '{taco_dir}' with prefix '{taco_name}'.")
        return

    for i, file_path in enumerate(tqdm(files_to_upload)):
        api.upload_file(
            path_or_fileobj=file_path,
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
    parser.add_argument(
        "--remove-existing",
        nargs="?",
        const="taco",
        default=None,
        metavar="SUFFIX",
        choices=["taco", "tacozip", "*"],
        help="Delete existing files from the HF repo before uploading. Defaults to 'taco' when used without a value; accepts 'tacozip' or '*' as well."
    )
    args = parser.parse_args()

    if args.dataset == "deepextremecubes":
        # This taco_path should be the specific .tacozip file or the directory containing it.
        taco_path = Path("/data/databases/deepextremecubes-video-final") 
        taco_name = "deepextremecubes-video-final" # This will be used if taco_path is a directory
        repo_id = "isp-uv-es/DeepExtremeCubes-video"
        if args.remove_existing is not None:
            remove_existing_files(repo_id, args.remove_existing)
        upload_taco(taco_path, taco_name, repo_id, file_extension="tacozip")

    elif args.dataset == "dynamicearthnet":
        taco_dir = Path("/data/databases/dynamicearthnet-video-final")
        repo_id = "isp-uv-es/DynamicEarthNet-video"
        if args.remove_existing is not None:
            remove_existing_files(repo_id, args.remove_existing)
        for version in ["71psnr", "77psnr"]:
            taco_name = f"dynamicearthnet-video-{version}"
            print(f"\nUploading {taco_name} → {repo_id}")
            upload_taco(taco_dir, taco_name, repo_id, file_extension="tacozip")

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