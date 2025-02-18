import json
import shutil
import datetime
from pathlib import Path
import boto3

from huggingface_hub import snapshot_download

import click

S3_API_URL = "https://1afbe4656a6b40d982ab5e730a39f6b9.r2.cloudflarestorage.com"

@click.command(help="Uploads the data from huggingface to an S3 bucket")
@click.argument("hf_repo_id", type=str)
@click.argument("s3_path", type=str)
@click.option("--bucket_name", type=str, default="datalab")
@click.option("--access_key_id", type=str, default="<access_key_id>")
@click.option("--access_key_secret", type=str, default="<access_key_secret>")
def main(hf_repo_id: str, s3_path: str, bucket_name: str, access_key_id: str, access_key_secret: str):
    curr_date = datetime.datetime.now().strftime("%Y_%m_%d")
    s3_path = f"{s3_path}/{curr_date}"

    download_folder = snapshot_download(repo_id=hf_repo_id)
    download_folder = Path(download_folder)
    contained_files = list(download_folder.glob("*"))
    contained_files = [f.name for f in contained_files] # Just get the base name
    manifest_file = download_folder / "manifest.json"

    with open(manifest_file, "w") as f:
        json.dump({"files": contained_files}, f)

    # Upload the files to S3
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_API_URL,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_key_secret,
        region_name="enam"
    )

    # Iterate through all files in the folder
    for file_path in download_folder.glob('*'):
        s3_key = f"{s3_path}/{file_path.name}"

        try:
            s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key
            )
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")

    shutil.rmtree(download_folder)

    print(f"Uploaded files to {s3_path}")

if __name__ == "__main__":
    main()



