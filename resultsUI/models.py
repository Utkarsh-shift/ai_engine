from django.db import models
import uuid

# class BatchEntry(models.Model):
#     batch_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
#     status = models.CharField(max_length=20, choices=[('pending', 'Pending'), ('processing', 'Processing'), ('processed', 'Processed')], default='pending')
#     created_at = models.DateTimeField(auto_now_add=True)

class BatchEntry(models.Model):
    batch_id = models.UUIDField( editable=False, unique=True)
    status = models.CharField(max_length=20, choices=[('pending', 'Pending'), ('processing', 'Processing'), ('processed', 'Processed')], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    results = models.JSONField(null=True, blank=True)



class LinkEntry(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    ]
    
    link = models.URLField(max_length=3000)
    
    unique_id = models.UUIDField( editable=False, unique=True)
    batch = models.ForeignKey(BatchEntry, to_field='batch_id', on_delete=models.CASCADE, related_name='links')
    status = models.TextField( choices=STATUS_CHOICES, default='pending')
    video_path = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.unique_id}: {self.link} [{self.get_status_display()}]"
