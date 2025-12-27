"""Views for stormwater analysis application."""

import logging
import os
import tempfile

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views import View
from django.views.generic import TemplateView

from .forms import SWMMModelForm
from .models import CalculationSession, SWMMModel
from .services import AnalysisService, CalculationRetrievalService, NavigationService

TEST_FILE = "/Users/rafalbuczynski/Git/stormwater-analysis/models/recomendations/wroclaw_pipes.inp"

logger = logging.getLogger(__name__)


# =============================================================================
# Mixins
# =============================================================================


class SessionOwnerMixin:
    """Mixin ensuring access only to user's own completed sessions."""

    def get_session(self):
        """Get session owned by current user."""
        return get_object_or_404(
            CalculationSession,
            id=self.kwargs["session_id"],
            user=self.request.user,
        )

    def validate_completed_or_redirect(self, session):
        """Check if session is completed. Returns redirect response or None."""
        if session.status != "completed":
            messages.warning(self.request, f"Analysis session is {session.status}.")
            return redirect("sa:analysis")
        return None


# =============================================================================
# Simple Views
# =============================================================================


class IndexView(TemplateView):
    """Home page view."""

    template_name = "sa/index.html"


class AboutView(TemplateView):
    """About page view."""

    template_name = "sa/about.html"


# =============================================================================
# Analysis Views
# =============================================================================


class AnalysisView(LoginRequiredMixin, View):
    """View for SWMM file analysis."""

    login_url = "/accounts/login/"
    template_name = "sa/analysis.html"

    def get(self, request):
        """Display analysis form."""
        return render(request, self.template_name, {"swmm_form": SWMMModelForm()})

    def post(self, request):
        """Handle file upload and analysis."""
        if request.POST.get("use_test_file") == "true":
            return self._handle_test_file(request)
        return self._handle_uploaded_file(request)

    def _handle_test_file(self, request):
        """Process test file analysis."""
        try:
            swmm_model = SWMMModel.objects.create(user=request.user, file=TEST_FILE, zone=0.8)
            session, success = AnalysisService.analyze_file(
                user=request.user,
                file_path=TEST_FILE,
                frost_zone=0.8,
                swmm_model=swmm_model,
            )

            if success:
                messages.success(request, "Test file analysis completed successfully!")
                return self._render_success(request, session)

            messages.error(request, "Error occurred while processing test file.")

        except Exception as e:
            logger.error(f"Test file analysis error: {e}")
            messages.error(request, f"Error occurred while analyzing test file: {e}")

        return self._render_form(request)

    def _handle_uploaded_file(self, request):
        """Process uploaded file analysis."""
        form = SWMMModelForm(request.POST, request.FILES)

        if not form.is_valid():
            return self._render_form(request, form)

        instance = form.save(commit=False)
        instance.user = request.user
        uploaded_file = request.FILES["file"]

        temp_path = self._create_temp_file(uploaded_file)
        instance.file = uploaded_file.name
        instance.save()

        frost_zone = instance.get_frost_zone_value()

        try:
            session, success = AnalysisService.analyze_file(
                user=request.user,
                file_path=temp_path,
                frost_zone=frost_zone,
                swmm_model=instance,
            )

            if success:
                messages.success(request, "Analysis completed successfully!")
                return self._render_success(request, session)

            messages.error(request, "Error occurred while saving calculation results.")

        finally:
            self._cleanup_temp_file(temp_path)

        return self._render_form(request)

    def _render_form(self, request, form=None):
        """Render analysis form."""
        return render(request, self.template_name, {"swmm_form": form or SWMMModelForm()})

    def _render_success(self, request, session):
        """Render successful analysis results."""
        data = CalculationRetrievalService.format_session_data_for_template(session)
        return render(
            request,
            self.template_name,
            {
                "swmm_form": SWMMModelForm(),
                "data": data,
                "session": session,
                "show_results": True,
            },
        )

    @staticmethod
    def _create_temp_file(uploaded_file) -> str:
        """Create temporary file from uploaded file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".inp")
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name

    @staticmethod
    def _cleanup_temp_file(path: str):
        """Remove temporary file."""
        try:
            os.unlink(path)
        except OSError:
            pass


class AnalysisResultsView(LoginRequiredMixin, SessionOwnerMixin, TemplateView):
    """View for displaying analysis results."""

    login_url = "/accounts/login/"
    template_name = "sa/analysis_results.html"

    def get(self, request, *args, **kwargs):
        session = self.get_session()

        redirect_response = self.validate_completed_or_redirect(session)
        if redirect_response:
            return redirect_response

        data = CalculationRetrievalService.format_session_data_for_template(session)
        return render(request, self.template_name, {"session": session, "data": data})


# =============================================================================
# Detail Views
# =============================================================================


class BaseElementDetailView(LoginRequiredMixin, SessionOwnerMixin, TemplateView):
    """Base view for element details (conduit/node/subcatchment)."""

    login_url = "/accounts/login/"
    template_name = "sa/detail.html"
    element_type: str = None
    element_name_kwarg: str = None

    def get_element(self, session):
        """Get the element from session. Override in subclasses."""
        raise NotImplementedError

    def get_related_objects(self, session, element):
        """Get related objects for the element. Override in subclasses."""
        return {}

    def get(self, request, *args, **kwargs):
        session = self.get_session()

        redirect_response = self.validate_completed_or_redirect(session)
        if redirect_response:
            return redirect_response

        element = self.get_element(session)
        if not element:
            element_name = kwargs[self.element_name_kwarg]
            messages.error(request, f"{self.element_type.title()} '{element_name}' not found in this session.")
            return redirect("sa:analysis_results", session_id=session.id)

        navigation = NavigationService.get_navigation_context(
            session,
            self.element_type,
            kwargs[self.element_name_kwarg],
        )

        context = {
            "session": session,
            self.element_type: element,
            "data_type": self.element_type,
            "navigation": navigation,
            **self.get_related_objects(session, element),
        }

        return self.render_to_response(context)


class ConduitDetailView(BaseElementDetailView):
    """Detailed view for a specific conduit."""

    element_type = "conduit"
    element_name_kwarg = "conduit_name"

    def get_element(self, session):
        return session.conduits.filter(conduit_name=self.kwargs["conduit_name"]).first()

    def get_related_objects(self, session, conduit):
        return {
            "inlet_node": session.nodes.filter(node_name=conduit.inlet_node).first(),
            "outlet_node": session.nodes.filter(node_name=conduit.outlet_node).first(),
            "subcatchment": session.subcatchments.filter(subcatchment_name=conduit.subcatchment).first(),
        }


class NodeDetailView(BaseElementDetailView):
    """Detailed view for a specific node."""

    element_type = "node"
    element_name_kwarg = "node_name"

    def get_element(self, session):
        return session.nodes.filter(node_name=self.kwargs["node_name"]).first()

    def get_related_objects(self, session, node):
        related_conduits = session.conduits.filter(Q(inlet_node=node.node_name) | Q(outlet_node=node.node_name))
        subcatchment_names = related_conduits.values_list("subcatchment", flat=True).distinct()

        return {
            "related_conduits": related_conduits,
            "related_subcatchments": session.subcatchments.filter(subcatchment_name__in=subcatchment_names),
        }


class SubcatchmentDetailView(BaseElementDetailView):
    """Detailed view for a specific subcatchment."""

    element_type = "subcatchment"
    element_name_kwarg = "subcatchment_name"

    def get_element(self, session):
        return session.subcatchments.filter(subcatchment_name=self.kwargs["subcatchment_name"]).first()

    def get_related_objects(self, session, subcatchment):
        related_conduits = session.conduits.filter(subcatchment=subcatchment.subcatchment_name)
        inlet_nodes = related_conduits.values_list("inlet_node", flat=True)
        outlet_nodes = related_conduits.values_list("outlet_node", flat=True)
        all_nodes = set(inlet_nodes) | set(outlet_nodes)

        return {
            "related_conduits": related_conduits,
            "related_nodes": session.nodes.filter(node_name__in=all_nodes),
        }


# =============================================================================
# History View
# =============================================================================


class HistoryView(LoginRequiredMixin, TemplateView):
    """View for user's calculation history."""

    login_url = "/accounts/login/"
    template_name = "sa/history.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        sessions = CalculationRetrievalService.get_user_sessions(self.request.user)

        total = len(sessions)
        completed = sum(1 for s in sessions if s.status == "completed")
        failed = sum(1 for s in sessions if s.status == "failed")
        processing = sum(1 for s in sessions if s.status == "processing")

        context.update(
            {
                "sessions": sessions,
                "stats": {
                    "total_sessions": total,
                    "completed_sessions": completed,
                    "failed_sessions": failed,
                    "processing_sessions": processing,
                    "completed_percentage": round((completed * 100 / total) if total > 0 else 0),
                    "failed_percentage": round((failed * 100 / total) if total > 0 else 0),
                    "processing_percentage": round((processing * 100 / total) if total > 0 else 0),
                    "total_conduits": sum(s.conduits.count() for s in sessions if s.status == "completed"),
                    "total_nodes": sum(s.nodes.count() for s in sessions if s.status == "completed"),
                    "total_subcatchments": sum(s.subcatchments.count() for s in sessions if s.status == "completed"),
                },
                "latest_session": sessions[0] if sessions else None,
            }
        )

        return context


# =============================================================================
# AJAX Views
# =============================================================================


class DeleteSessionView(LoginRequiredMixin, SessionOwnerMixin, View):
    """AJAX endpoint for deleting a calculation session. Returns JSON response."""

    login_url = "/accounts/login/"

    def post(self, request, *args, **kwargs):
        session = self.get_session()
        session_id = session.id

        try:
            session.delete()
            logger.info(f"Successfully deleted session {session_id}")
            return JsonResponse({"success": True, "message": f"Session {session_id} deleted successfully."})
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return JsonResponse(
                {"success": False, "message": "An error occurred while deleting the session."},
                status=500,
            )
