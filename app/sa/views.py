import logging
import os

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.timezone import now
from pyswmm import Simulation

from sa.core.data import DataManager
from sa.core.tests import TEST_FILE
from .forms import SWMMModelForm
from .models import CalculationSession
from .services import CalculationPersistenceService, CalculationRetrievalService

# TEST_FILE = "/Users/rafalbuczynski/Git/stormwater-analysis/models/recomendations/dataset/proba/generated/01_recomendation_template_D300_EulerExtreme120_S1_0_rural.inp"
FILE = "/Users/rafalbuczynski/Git/stormwater-analysis/models/recomendations/wroclaw_pipes.inp"

with Simulation(FILE) as sim:
    for step in sim:
        pass


logger = logging.getLogger(__name__)


def index(request):
    return render(request, "sa/index.html")


def about(request):
    return render(request, "sa/about.html")


def unique_filename(file):
    filename_base, filename_ext = os.path.splitext(file.name)
    unique_sufix = now().strftime("%Y%m%d%H%M%S%f")
    new_filename = f"{filename_base}_{unique_sufix}{filename_ext}"
    file_path = os.path.join(settings.MEDIA_ROOT, "user_models", new_filename)
    return file_path, new_filename


def save_uploaded_file(file_path, uploaded_file):
    """
    Saves an uploaded file to disk.

    Args:
        file_path (str): The path where the file should be saved.
        uploaded_file (UploadedFile): The uploaded file object obtained from a Django form.
    """
    with open(file_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)


@login_required(login_url="/accounts/login/")
def analysis(request):
    if request.method == "POST":
        swmm_form = SWMMModelForm(request.POST, request.FILES)
        if swmm_form.is_valid():
            instance = swmm_form.save(commit=False)
            instance.user = request.user
            uploaded_file = request.FILES["file"]

            file_path, new_filename = unique_filename(uploaded_file)
            save_uploaded_file(file_path, uploaded_file)

            instance.file = new_filename
            instance.save()

            # Create calculation session using frost zone from the model
            frost_zone = instance.get_frost_zone_value()
            session = CalculationPersistenceService.create_session(
                user=request.user,
                swmm_model=instance,
                frost_zone=frost_zone
            )

            try:
                # Run calculations with DataManager
                with DataManager(file_path, zone=frost_zone) as model:
                    # Save results to database
                    success = CalculationPersistenceService.save_calculation_results(session, model)
                    
                    if success:
                        messages.success(request, "Analysis completed successfully!")
                        return redirect('sa:analysis_results', session_id=session.id)
                    else:
                        messages.error(request, "Error occurred while saving calculation results.")
                        
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                session.status = 'failed'
                session.error_message = str(e)
                session.save()
                messages.error(request, f"Error occurred while performing calculations: {str(e)}")
            
            return render(request, "sa/analysis.html", {"swmm_form": swmm_form})
    else:
        # GET request - show demo data using FILE
        swmm_form = SWMMModelForm()
        with DataManager(FILE, zone=0.8) as model:
            print(model.dfc.columns)
            feature_columns = [ 
                'ValMaxFill', 'ValMaxV',
                'ValMinV', 'ValMaxSlope', 'ValMinSlope',
                'ValDepth', 'ValCoverage',
                'isMinDiameter', 'IncreaseDia', 'ReduceDia',
                "IncreaseSlope", "ReduceSlope",
                'NRoughness',
                'NMaxV', 'NInletDepth', 'NOutletDepth', 'NFilling', 'NMaxQ',
                'NInletGroundCover', 'NOutletGroundCover', "NSlope", 
                'InletNode', 'OutletNode',
                'Subcatchment', 
                'SbcCategory',
                "recommendation",
                "Geom1",
                "MaxV",
                "SlopePerMile",
                "MinRequiredSlope",
                "MaxAllowableSlope",
                "Filling",
                "InletGroundElevation",
                "InletNodeInvert",
                "InletGroundCover",
                "InletMaxDepth",
                "OutletGroundElevation",
                "OutletNodeInvert",
                "OutletGroundCover",
                "OutletMaxDepth",
                "MinDiameter",
            ]
            dfc = model.dfc[feature_columns]
            # dfc.to_excel("recomendations_output.xlsx")
            
            conduits_dict = dfc.reset_index().to_dict("records")
            nodes_dict = model.dfn.reset_index().to_dict("records")
            subcatchments_dict = model.dfs.reset_index().to_dict("records")

            formatted_dataset_names = {
                "conduits_data": "Conduits Data",
                "nodes_data": "Nodes Data",
                "subcatchments_data": "Subcatchments Data",
            }
            data = {
                formatted_dataset_names[key]: value
                for key, value in [
                    ("conduits_data", conduits_dict),
                    ("nodes_data", nodes_dict),
                    ("subcatchments_data", subcatchments_dict),
                ]
            }
            return render(request, "sa/analysis.html", {"swmm_form": swmm_form, "data": data})


@login_required(login_url="/accounts/login/")
def analysis_results(request, session_id):
    """Display results for a specific calculation session."""
    session = get_object_or_404(CalculationSession, id=session_id, user=request.user)
    
    if session.status != 'completed':
        messages.warning(request, f"Analysis session is {session.status}.")
        return redirect('sa:analysis')
    
    # Format data for template
    data = CalculationRetrievalService.format_session_data_for_template(session)
    
    return render(request, "sa/analysis_results.html", {
        "session": session,
        "data": data
    })


@login_required(login_url="/accounts/login/")
def history(request):
    """Display user's calculation history."""
    sessions = CalculationRetrievalService.get_user_sessions(request.user)
    
    return render(request, "sa/history.html", {
        "sessions": sessions
    })
