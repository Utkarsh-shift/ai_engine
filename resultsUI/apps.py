from django.apps import AppConfig
import sys
import logging

class ResultsuiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'resultsUI'

    def ready(self): 
        if 'runserver' in sys.argv or 'celery' in sys.argv: 
            try : 
                from django_celery_beat.models import PeriodicTask, IntervalSchedule
                schedule, created = IntervalSchedule.objects.get_or_create(
                    every=24,
                    period=IntervalSchedule.MINUTES,
                )
                 # Create or update the periodic task
                PeriodicTask.objects.update_or_create(
                    name='Check and shutdown if idle',
                    task='resultsUI.tasks.scheduled_shutdown_check',  # Make sure this is correct
                    defaults={'interval': schedule},
                )
                print("✅ Scheduled shutdown task registered.")
            except Exception as e:
                logging.error(f"❌ Failed to register periodic task: {e}")
