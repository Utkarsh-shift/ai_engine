import boto3
from botocore.exceptions import NoCredentialsError
import requests,time,os
import traceback
from botocore.config import Config
s3_config = Config(s3={'addressing_style': 'virtual'}) 
headers = {
        "Accept-Encoding": "identity"  # Avoid gzip corruption
    }

def generate_presigned_url_function(bucket_name, object_key, expiration=3600, access_key=None, secret_key=None, session_token=None):
    try:
        
        if access_key and secret_key:
            s3_client = boto3.client(
                's3',
                region_name='ap-south-1',
                config=s3_config,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token
            )
        else:
            s3_client = boto3.client('s3', region_name='ap-south-1', config=s3_config)

        metadata = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        print(f"Object exists. Size: {metadata['ContentLength']} bytes")

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )

        print(f"Presigned URL generated:\n{presigned_url}\n")
        return presigned_url

    except Exception as e:
        traceback.print_exc() 
        print(f" Failed to generate presigned URL: {e}")
        return None

import requests
import os
import traceback

# def download_file(url, output_path, headers=None):
#     try:
#         if headers is None:
#             headers = {}  # fallback if not provided

#         with requests.get(url, headers=headers, stream=True, timeout=60) as response:
#             response.raise_for_status()

#             # Check if content type is suspicious (e.g., an error page instead of a file)
#             content_type = response.headers.get('Content-Type', '')
#             if "html" in content_type.lower():
#                 raise ValueError("Received HTML instead of expected content. Possible 404 or error response.")

#             # Write content to file
#             with open(output_path, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
#                 f.flush()
#                 os.fsync(f.fileno())

#         print(f" Downloaded file saved to: {output_path}")
#         print(f" File size: {os.path.getsize(output_path)} bytes")

#     except Exception as e:
#         traceback.print_exc()
#         print(f" Download failed: {e}")

import subprocess
import shlex
import traceback

def download_file(url, output_path , header = None):
    try:
        # Safely build the wget command
        command = f"wget \"{url}\" -O \"{output_path}\""
        print(f"Running command: {command}")
        
        # Use shlex.split to handle spaces/special characters properly
        subprocess.run(shlex.split(command), check=True)

        print(f" Downloaded successfully to: {output_path}")
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        print(f" Download failed with exit code {e.returncode}")
    except Exception as e:
        traceback.print_exc()
        print(f" Download failed: {e}")


