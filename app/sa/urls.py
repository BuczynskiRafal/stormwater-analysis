from django.urls import path

from .views import (
    AboutView,
    AnalysisResultsView,
    AnalysisView,
    BulkDeleteSessionsView,
    ConduitDetailView,
    DeleteSessionView,
    HistoryView,
    IndexView,
    NodeDetailView,
    SubcatchmentDetailView,
)

app_name = "sa"

urlpatterns = [
    path("", IndexView.as_view(), name="index"),
    path("about/", AboutView.as_view(), name="about"),
    path("analysis/", AnalysisView.as_view(), name="analysis"),
    path("session/<int:session_id>/delete/", DeleteSessionView.as_view(), name="delete_session"),
    path("sessions/bulk-delete/", BulkDeleteSessionsView.as_view(), name="bulk_delete_sessions"),
    path("analysis/results/<int:session_id>/", AnalysisResultsView.as_view(), name="analysis_results"),
    path("analysis/results/<int:session_id>/conduit/<str:conduit_name>/", ConduitDetailView.as_view(), name="conduit_detail"),
    path("analysis/results/<int:session_id>/node/<str:node_name>/", NodeDetailView.as_view(), name="node_detail"),
    path(
        "analysis/results/<int:session_id>/subcatchment/<str:subcatchment_name>/",
        SubcatchmentDetailView.as_view(),
        name="subcatchment_detail",
    ),
    path("history/", HistoryView.as_view(), name="history"),
]
