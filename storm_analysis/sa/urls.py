from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from .views import upload_swmm_file

app_name = "sa"

urlpatterns = [
    path("upoload/", upload_swmm_file, name="upload")
]
