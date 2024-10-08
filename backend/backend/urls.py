from django.urls import path
from csv_processor.views import process_csv

urlpatterns = [
    path('process-csv/', process_csv, name='process_csv'),
    path('music-engine/', process_csv, name='generate_music_recommendations'),
]
