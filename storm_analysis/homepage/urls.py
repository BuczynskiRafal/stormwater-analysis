from django.urls import path
from homepage.views import about, index

app_name = "homepage"

urlpatterns = [
    path("", index, name="index"),
    path("about/", about, name="about"),
]
