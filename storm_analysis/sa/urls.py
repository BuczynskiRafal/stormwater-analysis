from django.urls import path

from .views import analysis, history

app_name = "sa"

urlpatterns = [
    path("analysis/", analysis, name="analysis"),
    path("history/", history, name="history"),
]
