import logging
import os

import swmmio
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils.timezone import now
from pyswmm import Simulation

from .forms import SWMMModelForm

logger = logging.getLogger(__name__)


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


@login_required(login_url="/login")
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

            # compute swmm file
            try:
                swmmio_model = swmmio.Model(file_path)
                with Simulation(swmmio_model.inp.path) as sim:
                    for _ in sim:
                        pass
                data = swmmio_model.subcatchments.dataframe.to_dict("records")

                return render(request, "sa/analysis.html", {"swmm_form": swmm_form, "data": data})
            except Exception as e:
                logger.error(e)
                messages.error(
                    request,
                    "Error occurred while performing calculations: {}".format(str(e)),
                )
            return render(request, "sa/analysis.html", {"swmm_form": swmm_form})
    else:
        swmm_form = SWMMModelForm()

    return render(request, "sa/analysis.html", {"swmm_form": swmm_form})


@login_required(login_url="/login")
def history(request):
    return render(request, "sa/history.html")
