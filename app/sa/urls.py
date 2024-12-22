from django.urls import path

from .views import about, analysis, history, index

app_name = "sa"

urlpatterns = [
    path("", index, name="index"),
    path("about/", about, name="about"),
    path("analysis/", analysis, name="analysis"),
    path("history/", history, name="history"),
]
