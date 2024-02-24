from django.urls import path
from django.conf import settings
from djan.conf.urls.static import static
from sa.view import upload_swmm_file

urlpatterns = [
    path("upoload/", upload_swmm_file, name="upload")
]