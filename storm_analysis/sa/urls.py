from django.urls import path

from .views import analysis

app_name = "sa"

urlpatterns = [
    path("", analysis, name="analysis")
]
