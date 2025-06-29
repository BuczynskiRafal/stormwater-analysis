from django.urls import path

from .views import about, analysis, analysis_results, history, index

app_name = "sa"

urlpatterns = [
    path("", index, name="index"),
    path("about/", about, name="about"),
    path("analysis/", analysis, name="analysis"),
    path("analysis/results/<int:session_id>/", analysis_results, name="analysis_results"),
    path("history/", history, name="history"),
]
