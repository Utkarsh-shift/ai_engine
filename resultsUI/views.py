# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .serializers import LinkSerializer, LinkEntrySerializer, BatchEntrySerializer, BatchSerializer
# from .models import LinkEntry, BatchEntry
# from .task import download_video, process_batch
# import os
# import uuid
# import requests
# from rest_framework.permissions import IsAuthenticated
# from rest_framework_simplejwt.authentication import JWTAuthentication

# class WebhookReceiverView(APIView):
#     def post(self, request, format=None):
#         # Access the JSON data from the request
#         data = request.data

#         # Log or process the received data
#         print("Webhook received:", data)

#         # Example: Extract specific fields from the JSON
#         event = data.get('event')
#         batch_id = data.get('batch_id')
#         batch_data = data.get('data')

#         # Perform any necessary processing or save to the database
#         # Example:
#         # process_event(event, batch_id, batch_data)

#         # Respond to the webhook sender
#         return Response({'status': 'received', 'event': event,"batch_id":batch_id,"data":batch_data}, status=status.HTTP_200_OK)
    

# class LinkEntryAPIView(APIView):
#     def post(self, request, *args, **kwargs):
#         serializer = LinkSerializer(data=request.data)

#         if serializer.is_valid():
#             links = serializer.validated_data['links']
#             batch_entry = BatchEntry.objects.create()
#             response_data = []

#             for link in links:
#                 link_entry = LinkEntry.objects.create(link=link, batch=batch_entry)
#                 video_path = download_video(link_entry, batch_entry.batch_id)
#                 link_entry.video_path = video_path if video_path else ''
#                 link_entry.status = 'pending' if video_path else 'failed'
#                 link_entry.save()
#                 entry_data = LinkEntrySerializer(link_entry).data
#                 response_data.append(entry_data)

#             process_batch.delay(batch_entry.batch_id)
#             batch_serializer = BatchEntrySerializer(batch_entry)
#             return Response(batch_serializer.data, status=status.HTTP_201_CREATED)
        
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     def update_batch_status(self, batch_entry):
#         if not batch_entry.links.exclude(status='processed').exists():
#             batch_entry.status = 'processed'
#         else:
#             batch_entry.status = 'processing'
#         batch_entry.save()

# class BatchResultView(APIView):
#     def get(self, request, batch_id, format=None):
#         try:
#             batch = BatchEntry.objects.get(batch_id=batch_id)
#             if batch.status == 'processed':
#                 serializer = BatchSerializer(batch)
#                 self.trigger_webhook(batch_id, serializer.data)
#                 return Response(serializer.data)
#             else:
#                 return Response({'error': 'Batch not processed yet'}, status=status.HTTP_400_BAD_REQUEST)
#         except BatchEntry.DoesNotExist:
#             return Response({'error': 'Batch not found'}, status=status.HTTP_404_NOT_FOUND)
        
    # def trigger_webhook(self, batch_id, data):
    #     webhook_url = 'http://127.0.0.1:8000/workbook'
    #     payload = {
    #         'batch_id': batch_id,
    #         'data': data,
    #         'event': 'batch_processed',
    #     }
    #     try:
    #         response = requests.post(webhook_url, json=payload)
    #         response.raise_for_status()
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error sending webhook: {e}")




from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import LinkSerializer, LinkEntrySerializer
from rest_framework.permissions import IsAuthenticated
from .task import download_video
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
from .serializers import LinkSerializer, LinkEntrySerializer, BatchEntrySerializer
from .models import LinkEntry, BatchEntry
# from .task import download_video
import uuid
from decouple import config

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import LinkSerializer, LinkEntrySerializer, BatchEntrySerializer,BatchSerializer
from .models import LinkEntry, BatchEntry
from .task import download_video
import uuid

from django.views.decorators.csrf import csrf_exempt
import json
import dotenv
endpoint_name=os.getenv("AWS_ENDPOINT")
region_name=os.getenv("AWS_REGION")
# you can use it in the main view directly by start_sagemaker_endpoint_task.delay('your-endpoint-name', 'your-region') this is only for testing 
def start_sagemaker_endpoint(request):
    start_sagemaker_endpoint_task.delay('your-endpoint-name', 'your-region')
    return JsonResponse({'status': 'SageMaker endpoint is being started asynchronously'}, status=202)


@csrf_exempt
def webhook(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request
            data = json.loads(request.body)
            print("Received data:", data)
 
            # Process the data as needed
            # e.g., validate, save to database, etc.
 
            return JsonResponse({'status': 'success', 'message': 'Webhook received successfully'}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)



class WebhookReceiverView(APIView):
    def post(self, request, format=None):
        # Access the JSON data from the request
        data = request.data

        # Log or process the received data
        print("Webhook received:", data)

        # Example: Extract specific fields from the JSON
        event = data.get('event')
        batch_id = data.get('batch_id')
        batch_data = data.get('data')

        # Perform any necessary processing or save to the database
        # Example:
        # process_event(event, batch_id, batch_data)

        # Respond to the webhook sender
        return Response({'status': 'received', 'event': event,"batch_id":batch_id,"data":batch_data}, status=status.HTTP_200_OK)
    


class LinkEntryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = LinkSerializer(data=request.data)
        print(request.data)
        if serializer.is_valid():
            # print(serializer.validated_data)
            links1 = serializer.validated_data
            # print(links1)
            new_links=[]
            Id=[]
            Questions=[]
            batch_id=str(request.data["batch_id"])
            webhook_url = str(request.data["server_url"])
            for item in links1["links"]:
                # print("ID:", item["id"])
                # print("Link:", item["link"])
                # Id.append(item["id"])
                new_links.append(item["link"])
                Questions.append(item["question"])

            
            # print("budyhucudyudgyugfygygyu",BatchEntry.objects.filter(batch_id=batch_id))
            # Retrieve all entries with the specified batch_id
            # try:
            # status_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('status', flat=True)
            # # except:
            # #     status_values=["not processed"]
            # print("**********************************",status_values[0])
            
            # if str(status_values[0])!="processed":
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

                batch_entry = BatchEntry.objects.create(batch_id=batch_id)

                response_data = []

                for item  in links1["links"]:
                    # Create a new LinkEntry with the BatchEntry instance
                    link_entry = LinkEntry.objects.create(link=item["link"],unique_id=item["id"], batch=batch_entry )
                    entry_data = LinkEntrySerializer(link_entry).data
                    response_data.append(entry_data)
                print(response_data)
                print("the url in signature post is" , webhook_url)
                process_batch.delay(batch_id=batch_entry.batch_id,Questions=Questions,webhook_url=webhook_url)
                # Serialize the batch entry and include it in the response
                batch_serializer = BatchEntrySerializer(batch_entry)
                # batch_id=batch_entry.batch_id
                # batch = BatchEntry.objecsts.get(batch_id=batch_id)
                # if batch.status == 'processed':
                #     serializer = BatchSerializer(batch)
                #     self.trigger_webhook(batch_id,batch_entry.results)
                response={"batch_id":batch_serializer.data["batch_id"],"status":batch_serializer.data["status"],"created_at":batch_serializer.data['created_at']}
                # batch_serializer[]
                return Response(response, status=status.HTTP_201_CREATED)
            # else:
            #     results_values = BatchEntry.objects.filter(batch_id=batch_id).values_list('results', flat=True)
            #     return Response(results_values,status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    def update_batch_status(self, batch_entry):
    # Check if all LinkEntry objects related to this batch have the status 'processed'
        if not batch_entry.links.exclude(status='processed').exists():
            # If all related LinkEntry objects are processed, update the BatchEntry status
            batch_entry.status = 'processed'
            batch_entry.save()
        else:
            # If not all are processed, set the batch status to processing
            batch_entry.status = 'processing'
            batch_entry.save()
###################################################Not Working ###################
    # def trigger_webhook(self, batch_id, data):
    #     signature_url="http://192.168.1.120:8020/get-signature"
    #     # try:
    #     signature=requests.get(signature_url)
    #     # except Exception as e:
    #     #     print("error getting signature",e)

    #     webhook_url = config('RESPONSE_FEEDBACK_URL')
    #     headers = {
    #         'Accept': 'application/json',  
    #         'Signature': signature
    #     }
    #     payload = {
    #         'batch_id': batch_id,
    #         'data': data,
    #         'event': 'batch_processed',
    #     }
    #     try:
    #         response = requests.post(webhook_url,headers=headers, json=payload)
    #         response.raise_for_status()
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error sending webhook: {e}")

# Other views remain the same as before

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
        webhook_url = 'http://127.0.0.1:8000/workbook'
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


