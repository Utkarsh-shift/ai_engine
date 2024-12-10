from celery import shared_task
from .models import LinkEntry, BatchEntry
from AIinterview import settings
import os,json,requests
import shutil
from .script.run_exp import show_results
# from openpyxl import load_workbook,Workbook
import boto3
import dotenv
# FUNCTION TO DOWNLOAD ALL THE VIDEOS FROM THE AWS 
from decouple import config
import botocore
import os,time
 
# Function to download a file from S3
def download_file_from_s3(bucket_name, object_key, local_path):

    
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"))
        # aws_session_token=os.getenv("SESSION_TOKEN")) # session token is optional
    try:

        s3.download_file(bucket_name, object_key, local_path)
        print(f'Successfully downloaded {object_key} from bucket {bucket_name} to {local_path}')
    except botocore.exceptions.ClientError as e:
        print(f'Error downloading file {object_key}: {e}')
 

def list_and_download_files(bucket_name, local_directory):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"))
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
 
        # Check if the bucket has any objects
        if 'Contents' in response:
            print(f'Files in bucket "{bucket_name}":')
            for obj in response['Contents']:
                object_key = obj['Key']
                print(f' - {object_key}')
                # Create local file path
                local_file_path = os.path.join(local_directory, object_key)
                # Create any necessary directories
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                # Download the file
                download_file_from_s3(bucket_name, object_key, local_file_path)
        else:
            print(f'Bucket "{bucket_name}" is empty.')
    except Exception as e:
        print(f'Error listing files: {e}')


def get_signature():
    # from dotenv import load_dotenv
    # load_dotenv()
    # os.getenv("SIGN_URL")
    # signature_url="http://192.168.1.127:8020/webhook/get-signature"
    signature_url=config('SIGN_URL')
    signature_res=requests.get(signature_url)
    sign_json=signature_res.json()
    real_signature=str(sign_json["data"])
    return real_signature



@shared_task
def trigger_webhook(batch_id, data):
    print("The code is in the webhook code $$$$$$$$$$$$$$$$$$$s")
    # signature_url="http://192.168.1.121:8020/webhook/get-signature"
    print("batch id -------------------------is ",batch_id)
# # try:
    # signature_res=requests.get(signature_url)
    # signature=signature_res.json()
# except Exception as e:
#     print("error getting signature",e)
    try:
        signature=get_signature()
        print("signature is ----------",signature)
    
        # webhook_url = "http://192.168.1.127:8020/webhook/video-response-ai-feedback"
        webhook_url=config('RESPONSE_FEEDBACK_URL')
        headers = {
        'Accept': 'application/json',  
        'Signature': signature
        }
        payload = {
                    'batch_id': str(batch_id),
                    'data': data,
                    'event': 'batch_processed',
                }
    
        print("Webhook url is -------------------------------",webhook_url)
        response = requests.post(webhook_url,headers=headers, json=payload)
    
    # while response.status_code!=200:
    #     time.sleep(3)
    #     response = requests.post(webhook_url,headers=headers, json=payload)
    
    # if response.status_code==200:
    #     print("succesfully request is send *******************************************************")
        print("json is given as --------------------------------------------------------------",response.json())
    except Exception as e:
        print("exception in webhhok part is ---------------------- ",e)

@shared_task
def start_sagemaker_endpoint_task(endpoint_name, region):
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    try:
        # Start the SageMaker endpoint
        response = sagemaker_client.describe_endpoint(EndpointName='your-endpoint-name')

        # Check if the endpoint is not already in service
        if response['EndpointStatus'] != 'InService':
            sagemaker_client.update_endpoint(
                EndpointName='your-endpoint-name',
                EndpointConfigName='your-endpoint-config-name'
            )
    except Exception as e:
        print(e)
        
@shared_task
def start_ec2_instance():
    """
    Start an EC2 instance with specified AWS credentials, if it is not already running.
 
    :param instance_id: The ID of the EC2 instance to start.
    :param aws_access_key_id: Your AWS access key ID.
    :param aws_secret_access_key: Your AWS secret access key.
    :param region_name: The AWS region where the instance is located (default: 'ap-south-1').
    """
    # Create a session using provided AWS credentials and region
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    
    ec2_client = session.client('ec2')
 
    try:
        # Describe the instance to check its state
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']
 
        if instance_state == 'running':
            print(f'Instance {instance_id} is already running.')
            return None
 
        # Start the instance
        start_response = ec2_client.start_instances(
            InstanceIds=[instance_id]
        )
        
        # Extract current and previous states from the response
        for instance in start_response['StartingInstances']:
            instance_id = instance['InstanceId']
            current_state = instance['CurrentState']
            previous_state = instance['PreviousState']
            
            print(f'Starting instance {instance_id}...')
            print(f'Current State: {current_state["Name"]}, Previous State: {previous_state["Name"]}')
 
        # Wait for the instance to be in the 'running' state
        ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
        
        # Reload the instance state after starting
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance_state = response['Reservations'][0]['Instances'][0]['State']['Name']
        print(f'Instance {instance_id} is now in {instance_state} state.')
 
    except Exception as e:
        print(f'Error starting instance {instance_id}: {e}')
 
def stop_ec2_instance():
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    
    ec2_client = session.client('ec2')
    response = ec2_client.stop_instances(
    InstanceIds=[
        'string',
    ],
    Hibernate=True|False,
    DryRun=True|False,
    Force=True|False
)


@shared_task
def process_batch(batch_id,Questions):
    print("The current batch id in process is givenas " , batch_id)
    try:
        print("The code is here in process_batch")
        
        batch_entry = BatchEntry.objects.get(batch_id=batch_id)
        link_entries = LinkEntry.objects.filter(batch=batch_entry).order_by('id')
        # processed_json = []
        processing_dir = os.path.join(settings.MEDIA_ROOT)
        os.makedirs(processing_dir, exist_ok=True)
        
        batch_folder_path = os.path.join(settings.MEDIA_DOWNLOAD, str(batch_entry.batch_id))
        os.makedirs(batch_folder_path,exist_ok=True)

        img_processing_dir=settings.FRAME_ROOT
        os.makedirs(img_processing_dir,exist_ok=True)
        # downloding checkpoints
        import dotenv
        from dotenv import load_dotenv
        load_dotenv()
        ondrive_user_name=os.getenv("ONEDRIVE_USER")
        ondrive_password=os.getenv("ONEDRIVE_PASS")
        #openAUgraph models
        


        path = os.getcwd()
        path = path + "/resultsUI/"
        if path.endswith("/resultsUI/resultsUI/"):
            path=path.replace("/resultsUI/resultsUI/","/resultsUI/")
        models_path = path+"/OpenGraphAU/checkpoints/"
        no_of_files = os.listdir(models_path)

        if len(no_of_files) > 2:
            print("Models already downloaded")
        else:
            model_urls = [
                'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            ]
            
            for url in model_urls:
                filename = url.split('/')[-1]
                file_path = os.path.join(models_path, filename)
                os.system(f"wget -O {file_path} {url}")
        # 2nd stage model
        path=path+"checkpoints/OpenGprahAU-ResNet50_second_stage.pth"
        if os.path.exists(path=path):
            print("Already downloaded the checkpoint")
        else:
            os.system(f"""wget -O {path} --user={ondrive_user_name} --password={ondrive_password} https://almabay-my.sharepoint.com/:u:/p/varun_kumar/EXKw_nvbUqhJrTLZZflt8SIBCpXubzGr3NucqisYjP_V3w?download=1""")
        print("link entries are -----------------------------------------------------------------",link_entries)
        for link_entry in link_entries:
            # Mark the current link entry as processing
            link_entry.status = 'processing'
            link_entry.save()
            video_filename = f"{link_entry.unique_id}.mp4"
            video_path = os.path.join(batch_folder_path , video_filename)
            # command = f"yt-dlp -f mp4 -o {video_path} {link_entry.link}"
            # command=os.system(f"""wget -O {video_path} --user={ondrive_user_name} --password={ondrive_password}  {link_entry.link} """)
            # response_code = os.system(command)
            import subprocess

            # Construct the command as a list
            print("The code is here in command ")
            command = [
            "wget",
            "-O", video_path,
            link_entry.link
            ]
            response_code = subprocess.run(command, check=True, capture_output=True, text=True)
            
            if response_code.returncode == 0 : 
                link_entry.video_path = video_path  # Store the absolute path
                link_entry.status = 'processing'
                link_entry.save()
                print(f"Video downloaded and saved to {video_path}")
            else:
                link_entry.status = 'failed'
                link_entry.save()
                print(f"Failed to download video with wget command. Response code: {response_code}")
                raise Exception(f"Failed to download video with wget command. Response code: {response_code}")
            # Move the video file to the processing directory
            video_filename = os.path.basename(link_entry.video_path)
            processing_video_path = os.path.join(processing_dir, video_filename)
            shutil.move(link_entry.video_path, processing_video_path)

            # Perform the processing (e.g., extract audio)
        result1= show_results(Questions)  # Assuming show_results processes the video and returns JSON
        print("print result 1 is here:",result1)
        # result1['ocean_values']=result1['ocean_values'].tolist()
        result = json.dumps(result1) 
        result=json.loads(result)
        # result=result1
        print("the type is -------------------------------------------------------",type(result)) # Used to Set the Json Response 
        #     processed_json.append(result)
        

        for link_entry in link_entries:
            link_entry.status = 'processed'
            link_entry.save()


        batch_entry.status = 'processed'
        batch_entry.results = result# Save the results as a JSON string
        batch_entry.save()

        trigger_webhook(batch_id,result)

    # except BatchEntry.DoesNotExist:
    #     print(f"BatchEntry with ID {batch_id} does not exist.")
    except Exception as e:

        link_entry.status = f"failed : {e}"
        link_entry.save()
        print(f"Exception occurred: {e}")
        webhook_url = 'http://127.0.0.1:5000/test'
        payload = {
            'batch_id': batch_id,
            'data': {"status":"failed"},
            'event': 'batch_processed',
        }
        response = requests.post(webhook_url, json=payload)
        print(f"Error sending webhook: {e}")
        raise Exception
        return None


def download_video(link_entry, video_path):
    try:
        command = f"yt-dlp -f mp4 -o {video_path} {link_entry.link}"
        response_code = os.system(command)
        if response_code == 0:
            link_entry.video_path = video_path 
            link_entry.status = 'processed'
            link_entry.save()
            print(f"Video downloaded and saved to {video_path}")
            return video_path
        else:
            link_entry.status = 'failed'
            link_entry.save()
            print(f"Failed to download video with wget command. Response code: {response_code}")
            return None
    except Exception as e:
        link_entry.status = 'failed'
        link_entry.save()
        print(f"Exception occurred: {e}")
        return None