from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
from resultsUI.views import LinkEntryAPIView,BatchResultView,WebhookReceiverView
from .libcode import TokenObtainPairView,TokenRefreshView



urlpatterns = [
    path('api/access_token',TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/link-entry/', LinkEntryAPIView.as_view(), name='link-entry'),
    path('api/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
  #  path('',views.Welcome),
    path('batch/<str:batch_id>/results/', BatchResultView.as_view(), name='batch-results'),
    path('api/interviewTest',LinkEntryAPIView.as_view()),
    path("workbook",WebhookReceiverView.as_view(), name='webhook-receiver'),
    path('webhook/', views.webhook, name='webhook'),
    path('wake_aws/',views.start_sagemaker_endpoint,name="wake_aws")
    ]