import os
import shutil
import boto3
from platformdirs import user_cache_dir
import time
import tempfile

def download_s3_directory(s3_url, local_dir):
    """Download an entire directory from an S3-compatible storage to a local directory."""
    s3 = boto3.client(
        "s3",
    )
    
    bucket, prefix = s3_url.replace("s3://", "").split("/", 1)

    # Use tempfile.TemporaryDirectory to automatically clean up
    with tempfile.TemporaryDirectory() as temp_dir:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    rel_path = obj["Key"].replace(prefix, "").lstrip("/")
                    local_file = os.path.join(temp_dir, rel_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    
                    # Download the file (no retry logic here)
                    s3.download_file(bucket, obj["Key"], local_file)

        shutil.move(temp_dir, local_dir)  # Atomic rename

def validate_download(local_path, expected_size=None):
    """Optionally validate if the download was successful (e.g., check size or hash)."""
    if not os.path.exists(local_path):
        print(f"File {local_path} does not exist.")
        return False

    return True


class S3Mixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path.startswith("s3://"):
            cache_dir = os.path.join(user_cache_dir('surya'), 'models')
            model_name = pretrained_model_name_or_path.rstrip("/").split("/")[-1]  # Extract model name from S3 path
            local_path = os.path.join(cache_dir, model_name)
            
            # Retry logic for downloading the model folder
            retries = 3
            delay = 5
            attempt = 0
            success = False
            while not success and attempt < retries:
                try:
                    if not os.path.exists(local_path):
                        print(f"Downloading model from {pretrained_model_name_or_path} to {local_path}...")
                        download_s3_directory(pretrained_model_name_or_path, local_path)
                        
                        # Optionally validate download
                        if not validate_download(local_path):
                            print(f"Validation failed for {pretrained_model_name_or_path}. Attempting to re-download.")
                            download_s3_directory(pretrained_model_name_or_path, local_path)  # Retry the download if validation fails
                    else:

                        print(f"Using cached model at {local_path}")
                    
                    success = True  # If download and validation succeed
                except Exception as e:
                    print(f"Error downloading model from {pretrained_model_name_or_path}. Attempt {attempt+1} of {retries}. Error: {e}")
                    attempt += 1
                    if attempt < retries:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)  # Wait before retrying
                    else:
                        print(f"Failed to download {pretrained_model_name_or_path} after {retries} attempts.")
                        raise e  # Reraise exception after max retries

            pretrained_model_name_or_path = local_path
        
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)