import logging
import os
import tempfile

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from pyswmm import Simulation

from sa.core.data import DataManager
from .forms import SWMMModelForm
from .models import CalculationSession, SWMMModel
from .services import CalculationPersistenceService, CalculationRetrievalService

TEST_FILE = "/Users/rafalbuczynski/Git/stormwater-analysis/models/recomendations/wroclaw_pipes.inp"

with Simulation(TEST_FILE) as sim:
    for step in sim:
        pass


logger = logging.getLogger(__name__)


def index(request):
    return render(request, "sa/index.html")


def about(request):
    return render(request, "sa/about.html")


def create_temp_file(uploaded_file):
    """
    Creates a temporary file from uploaded file.

    Args:
        uploaded_file (UploadedFile): The uploaded file object obtained from a Django form.

    Returns:
        str: Path to the temporary file
    """
    # Create a temporary file with .inp extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".inp")

    # Write the uploaded file content to the temporary file
    for chunk in uploaded_file.chunks():
        temp_file.write(chunk)

    temp_file.close()
    return temp_file.name


@login_required(login_url="/accounts/login/")
def analysis(request):
    if request.method == "POST":
        # Check if it's a TEST_FILE mock request
        if request.POST.get("use_test_file") == "true":
            # Use TEST_FILE for quick testing
            try:
                # Create mock SWMM model
                mock_swmm_model = SWMMModel.objects.create(user=request.user, file="test_wroclaw_pipes.inp", zone=0.8)

                # Create session
                session = CalculationPersistenceService.create_session(
                    user=request.user, swmm_model=mock_swmm_model, frost_zone=0.8
                )

                # Run analysis with TEST_FILE
                with DataManager(TEST_FILE, zone=0.8) as model:
                    success = CalculationPersistenceService.save_calculation_results(session, model)

                    if success:
                        # Format data for display with clickable links
                        data = CalculationRetrievalService.format_session_data_for_template(session)
                        messages.success(request, "Test file analysis completed successfully!")
                        return render(
                            request,
                            "sa/analysis.html",
                            {"swmm_form": SWMMModelForm(), "data": data, "session": session, "show_results": True},
                        )
                    else:
                        messages.error(request, "Error occurred while processing test file.")

            except Exception as e:
                logger.error(f"Test file analysis error: {e}")
                messages.error(request, f"Error occurred while analyzing test file: {str(e)}")

        else:
            # Regular file upload
            swmm_form = SWMMModelForm(request.POST, request.FILES)
            if swmm_form.is_valid():
                instance = swmm_form.save(commit=False)
                instance.user = request.user
                uploaded_file = request.FILES["file"]

                # Create temporary file instead of permanent storage
                temp_file_path = create_temp_file(uploaded_file)

                # Set a simple filename for the database record
                instance.file = uploaded_file.name
                instance.save()

                # Create calculation session using frost zone from the model
                frost_zone = instance.get_frost_zone_value()
                session = CalculationPersistenceService.create_session(
                    user=request.user, swmm_model=instance, frost_zone=frost_zone
                )

                try:
                    # Run calculations with DataManager using temporary file
                    with DataManager(temp_file_path, zone=frost_zone) as model:
                        # Save results to database
                        success = CalculationPersistenceService.save_calculation_results(session, model)

                        if success:
                            # Format data for display with clickable links
                            data = CalculationRetrievalService.format_session_data_for_template(session)
                            messages.success(request, "Analysis completed successfully!")
                            return render(
                                request,
                                "sa/analysis.html",
                                {"swmm_form": SWMMModelForm(), "data": data, "session": session, "show_results": True},
                            )
                        else:
                            messages.error(request, "Error occurred while saving calculation results.")

                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                    session.status = "failed"
                    session.error_message = str(e)
                    session.save()
                    messages.error(request, f"Error occurred while performing calculations: {str(e)}")

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except OSError:
                        pass

        return render(request, "sa/analysis.html", {"swmm_form": SWMMModelForm()})

    else:
        # GET request - show only form without any data processing
        swmm_form = SWMMModelForm()
        return render(request, "sa/analysis.html", {"swmm_form": swmm_form})


@login_required(login_url="/accounts/login/")
def analysis_results(request, session_id):
    """Display results for a specific calculation session."""
    session = get_object_or_404(CalculationSession, id=session_id, user=request.user)

    if session.status != "completed":
        messages.warning(request, f"Analysis session is {session.status}.")
        return redirect("sa:analysis")

    # Format data for template
    data = CalculationRetrievalService.format_session_data_for_template(session)

    return render(request, "sa/analysis_results.html", {"session": session, "data": data})


@login_required(login_url="/accounts/login/")
def history(request):
    """Display user's calculation history with detailed statistics."""
    sessions = CalculationRetrievalService.get_user_sessions(request.user)

    # Calculate statistics
    total_sessions = len(sessions)
    completed_sessions = sum(1 for s in sessions if s.status == "completed")
    failed_sessions = sum(1 for s in sessions if s.status == "failed")
    processing_sessions = sum(1 for s in sessions if s.status == "processing")

    # Calculate percentages
    completed_percentage = round((completed_sessions * 100 / total_sessions) if total_sessions > 0 else 0)
    failed_percentage = round((failed_sessions * 100 / total_sessions) if total_sessions > 0 else 0)
    processing_percentage = round((processing_sessions * 100 / total_sessions) if total_sessions > 0 else 0)

    # Calculate totals for completed sessions
    total_conduits = sum(s.conduits.count() for s in sessions if s.status == "completed")
    total_nodes = sum(s.nodes.count() for s in sessions if s.status == "completed")
    total_subcatchments = sum(s.subcatchments.count() for s in sessions if s.status == "completed")

    # Get most recent session details
    latest_session = sessions[0] if sessions else None

    context = {
        "sessions": sessions,
        "stats": {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "failed_sessions": failed_sessions,
            "processing_sessions": processing_sessions,
            "completed_percentage": completed_percentage,
            "failed_percentage": failed_percentage,
            "processing_percentage": processing_percentage,
            "total_conduits": total_conduits,
            "total_nodes": total_nodes,
            "total_subcatchments": total_subcatchments,
        },
        "latest_session": latest_session,
    }

    return render(request, "sa/history.html", context)


@login_required(login_url="/accounts/login/")
def conduit_detail(request, session_id, conduit_name):
    """Display detailed view for a specific conduit."""
    session = get_object_or_404(CalculationSession, id=session_id, user=request.user)

    if session.status != "completed":
        messages.warning(request, f"Analysis session is {session.status}.")
        return redirect("sa:analysis")

    # Get conduit data
    conduit_data = session.conduits.filter(conduit_name=conduit_name).first()
    if not conduit_data:
        messages.error(request, f"Conduit '{conduit_name}' not found in this session.")
        return redirect("sa:analysis_results", session_id=session_id)

    # Get related nodes
    inlet_node = session.nodes.filter(node_name=conduit_data.inlet_node).first()
    outlet_node = session.nodes.filter(node_name=conduit_data.outlet_node).first()

    # Get related subcatchment
    subcatchment = session.subcatchments.filter(subcatchment_name=conduit_data.subcatchment).first()

    context = {
        "session": session,
        "conduit": conduit_data,
        "inlet_node": inlet_node,
        "outlet_node": outlet_node,
        "subcatchment": subcatchment,
        "data_type": "conduit",
    }

    return render(request, "sa/detail.html", context)


@login_required(login_url="/accounts/login/")
def node_detail(request, session_id, node_name):
    """Display detailed view for a specific node."""
    session = get_object_or_404(CalculationSession, id=session_id, user=request.user)

    if session.status != "completed":
        messages.warning(request, f"Analysis session is {session.status}.")
        return redirect("sa:analysis")

    # Get node data
    node_data = session.nodes.filter(node_name=node_name).first()
    if not node_data:
        messages.error(request, f"Node '{node_name}' not found in this session.")
        return redirect("sa:analysis_results", session_id=session_id)

    # Get related conduits (where this node is inlet or outlet)
    related_conduits = session.conduits.filter(inlet_node=node_name).union(session.conduits.filter(outlet_node=node_name))

    context = {
        "session": session,
        "node": node_data,
        "related_conduits": related_conduits,
        "data_type": "node",
    }

    return render(request, "sa/detail.html", context)


@login_required(login_url="/accounts/login/")
def subcatchment_detail(request, session_id, subcatchment_name):
    """Display detailed view for a specific subcatchment."""
    session = get_object_or_404(CalculationSession, id=session_id, user=request.user)

    if session.status != "completed":
        messages.warning(request, f"Analysis session is {session.status}.")
        return redirect("sa:analysis")

    # Get subcatchment data
    subcatchment_data = session.subcatchments.filter(subcatchment_name=subcatchment_name).first()
    if not subcatchment_data:
        messages.error(request, f"Subcatchment '{subcatchment_name}' not found in this session.")
        return redirect("sa:analysis_results", session_id=session_id)

    # Get related conduits
    related_conduits = session.conduits.filter(subcatchment=subcatchment_name)

    context = {
        "session": session,
        "subcatchment": subcatchment_data,
        "related_conduits": related_conduits,
        "data_type": "subcatchment",
    }

    return render(request, "sa/detail.html", context)
