from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
# from resultsUI.Ai_Agent_Evaluation.run_first import start_evaluation
from AIinterview import settings
from django.http import HttpResponse, JsonResponse
from .models import LinkEntry
import os,requests
import uuid
from rest_framework_simplejwt.authentication import JWTAuthentication
from .task import process_batch,start_sagemaker_endpoint_task
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import LinkEntry, BatchEntry
from task import start_evaluation
from decouple import config
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import LinkSerializer, LinkEntrySerializer, BatchEntrySerializer,BatchSerializer
from .models import LinkEntry, BatchEntry


from django.views.decorators.csrf import csrf_exempt
import json
import dotenv
endpoint_name=os.getenv("AWS_ENDPOINT")
region_name=os.getenv("AWS_REGION")

def start_sagemaker_endpoint(request):
    start_sagemaker_endpoint_task.delay('your-endpoint-name', 'your-region')
    return JsonResponse({'status': 'SageMaker endpoint is being started asynchronously'}, status=202)


@csrf_exempt
def webhook(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)
            return JsonResponse({'status': 'success', 'message': 'Webhook received successfully'}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)



class WebhookReceiverView(APIView):
    def post(self, request, format=None):
        data = request.data
        print("Webhook received:", data)
        event = data.get('event')
        batch_id = data.get('batch_id')
        batch_data = data.get('data')
        return Response({'status': 'received', 'event': event,"batch_id":batch_id,"data":batch_data}, status=status.HTTP_200_OK)
    
def delete_files_in_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)  
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder {folder_path} does not exist!")
import json
class LinkEntryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        print( "**************** request.datarequest.datarequest.data",request.data)
        if isinstance(request.data, str):
            data = json.loads(request.data)
        else:
            data = request.data
        
        serializer = LinkSerializer(data=data)
        print( "**************** request.datarequest.datarequest.data",data)
        # delete_files_in_folder("/home/ubuntu/new_AVIPA/resultsUI/datasets/ChaLearn/test")
        print(serializer , "**************")
        if serializer.is_valid():

            if data['is_agent']=="1":
                links1 = serializer.validated_data
                

                webhook_url = str(data["server_url"])
                session_id=data['openai_id']
                batch_id=str(data["batch_id"])
                print("======================================",session_id)
                if BatchEntry.objects.filter(batch_id=batch_id).exists():
                    status_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('status', flat=True)
                    if str(status_values[0])=="processed":
                        results_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('results', flat=True)
                        result_final={"batch_id":batch_id,"status":"processed","data":results_values[0]}
                        return Response(result_final,status=status.HTTP_201_CREATED)
                    if str(status_values[0])=="pending":
                        results_values = BatchEntry.objects.filter(batch_id=batch_id).values()[0]
                        filtered_data = {key: value for key, value in results_values.items() if key != "id"}
                        filtered_data1 = {key: value for key, value in filtered_data.items() if key != "results"}
                        return Response(filtered_data1,status=status.HTTP_201_CREATED)
                else:
                    s3_link = data["links"][0].get("link")
                    batch_entry = BatchEntry.objects.create(batch_id=batch_id, request=data)
                    print("S3 Link:", s3_link)
                    skills=data["skill"]
                    focus_skills=data["focus_skill"]
                    proctoring_data = data["proctoring_data"]                 
                 
                    for item  in links1["links"]:
                        link_entry = LinkEntry.objects.create(link=item["link"],unique_id=item["id"], batch=batch_entry )
                       
                        
                    start_evaluation.delay(session_id,s3_link,skills,focus_skills,proctoring_data,webhook_url,batch_id)  # make changes here
                    batch_serializer = BatchEntrySerializer(batch_entry)
                    response={"batch_id":batch_serializer.data["batch_id"],"status":batch_serializer.data["status"],"created_at":batch_serializer.data['created_at']}
                    return Response(response, status=status.HTTP_201_CREATED)
       
            else:
                links1 = serializer.validated_data
                new_links=[]
                Questions=[]
                batch_id=str(data["batch_id"])
                webhook_url = str(data["server_url"])
                proctoring_data = data["proctoring_data"]
                skills=data["skill"]
                focus_skills=data["focus_skill"]
                print(skills,"---------------------------------------------------",focus_skills)
              
                print(proctoring_data,"........................")
                for item in links1["links"]:
                    new_links.append(item["link"])
                    Questions.append(item["question"])
                if BatchEntry.objects.filter(batch_id=batch_id).exists():
                    status_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('status', flat=True)
                    if str(status_values[0])=="processed":
                        results_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('results', flat=True)
                        result_final={"batch_id":batch_id,"status":"processed","data":results_values[0]}
                        return Response(result_final,status=status.HTTP_201_CREATED)
                    if str(status_values[0])=="pending":
                        results_values = BatchEntry.objects.filter(batch_id=batch_id).values()[0]
                        filtered_data = {key: value for key, value in results_values.items() if key != "id"}
                        filtered_data1 = {key: value for key, value in filtered_data.items() if key != "results"}
                        return Response(filtered_data1,status=status.HTTP_201_CREATED)
                else:
                    batch_entry = BatchEntry.objects.create(batch_id=batch_id, request=data)
                  
                    for item  in links1["links"]:
                        link_entry = LinkEntry.objects.create(link=item["link"],unique_id=item["id"], batch=batch_entry )
                        entry_data = LinkEntrySerializer(link_entry).data
                       
                    process_batch.delay(skills,focus_skills,proctoring_data,batch_id=batch_entry.batch_id,Questions=Questions,webhook_url=webhook_url)
                    batch_serializer = BatchEntrySerializer(batch_entry)
                    response={"batch_id":batch_serializer.data["batch_id"],"status":batch_serializer.data["status"],"created_at":batch_serializer.data['created_at']}
                    return Response(response, status=status.HTTP_201_CREATED)
                
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    def update_batch_status(self, batch_entry):
    # Check if all LinkEntry objects related to this batch have the status 'processed'
        if not batch_entry.links.exclude(status='processed').exists():
            # If all related LinkEntry objects are processed, update the BatchEntry status
            print(batch_entry.results , "There are the results of the code @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            if batch_entry.results is None:
                batch_entry.status = 'failed'
                batch_entry.save()

            else:
                batch_entry.status = 'processed'
                batch_entry.save()
        else:
            # If not all are processed, set the batch status to processing
            batch_entry.status = 'processing'
            batch_entry.save()

class BatchResultView(APIView):
    def get(self, request, batch_id, format=None):
        try:
            batch = BatchEntry.objects.get(batch_id=batch_id)
            if batch.status == 'processed':
                serializer = BatchSerializer(batch)
                self.trigger_webhook(batch_id, serializer.data)
                return Response(serializer.data)
            else:
                return Response({'error': 'Batch not processed yet'}, status=status.HTTP_400_BAD_REQUEST)
        except BatchEntry.DoesNotExist:
            return Response({'error': 'Batch not found'}, status=status.HTTP_404_NOT_FOUND)
    def trigger_webhook(self, batch_id, data):
        webhook_url = config('webhook_url')
        payload = {
            'batch_id': batch_id,
            'data': data,
            'event': 'batch_processed',
        }
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending webhook: {e}")
def ERROR_RESPONSE(request):
    response = {
        'error': 'Internal Server Error',
        'status_code': '500'
    }
    return JsonResponse(response)

def ERROR_404(request, exception):
    response = {
        'error': 'Not Found',
        'status_code': '404'
    }
    return JsonResponse(response, status=404)


