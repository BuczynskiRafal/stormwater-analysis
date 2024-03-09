from django.urls import path

from .views import upload_swmm_file

app_name = "sa"

urlpatterns = [
    path("", upload_swmm_file, name="upload")
]
