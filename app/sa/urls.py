from django.urls import path

from .views import about, analysis, analysis_results, history, index, conduit_detail, node_detail, subcatchment_detail

app_name = "sa"

urlpatterns = [
    path("", index, name="index"),
    path("about/", about, name="about"),
    path("analysis/", analysis, name="analysis"),
    path("analysis/results/<int:session_id>/", analysis_results, name="analysis_results"),
    path("analysis/results/<int:session_id>/conduit/<str:conduit_name>/", conduit_detail, name="conduit_detail"),
    path("analysis/results/<int:session_id>/node/<str:node_name>/", node_detail, name="node_detail"),
    path(
        "analysis/results/<int:session_id>/subcatchment/<str:subcatchment_name>/", subcatchment_detail, name="subcatchment_detail"
    ),
    path("history/", history, name="history"),
]
