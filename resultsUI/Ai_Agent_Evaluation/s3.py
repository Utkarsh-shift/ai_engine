import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_video_to_s3(file_name, bucket_name, object_name=None, aws_access_key_id=None, aws_secret_access_key=None):
    
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3', 
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    )  

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        
        
        print(f"File {file_name} uploaded successfully to {bucket_name}/{object_name}")
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        print(f"File URL: {file_url}")

        os.remove(file_name)
        print(f"File {file_name} deleted from local system.")

        return file_url
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    except NoCredentialsError:
        print("Credentials not available.")
        return None
    except PartialCredentialsError:
        print("Incomplete credentials.")
        return None
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

if __name__ == "__main__":
    file_name = 'testusername_1234_jobid12345.mp4'  
    bucket_name = 'ai-int-qa.placecom.co'  
    aws_access_key_id = 'AKIA4AS4R4TD5G4RMEX2' 
    aws_secret_access_key = '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'  
    video_url = upload_video_to_s3(file_name, bucket_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    if video_url:
        print(f"The video is available at: {video_url}")


#file_url= https://ai-int-qa.placecom.co.s3.amazonaws.com/C:\Users\ADMIN\Desktop\evaluation\MicrosoftTeams-video.mp4ṭ