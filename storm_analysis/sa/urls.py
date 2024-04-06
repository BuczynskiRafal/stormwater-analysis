from django.urls import path

from .views import analysis, history, upload_file

app_name = "sa"

urlpatterns = [
    path("analysis/upload/", upload_file, name="upload_file"),
    path("analysis/", analysis, name="analysis"),
    path("history/", history, name="history"),
]
