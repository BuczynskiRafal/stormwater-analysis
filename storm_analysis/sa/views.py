from django.shortcuts import render
from .forms import UserSWMMModelForm


def upload_swmm_file(request):
    if request.method == "POST":
        form = UserSWMMModelForm(request.POST, request.FILES)
        if form.is_valid():
            user_file = form.save()
        
            """
            Handle calculations and classifications
            """

            return render(request, "results.html", {"results": results})
    else:
        form = UserSWMMModelForm()
    return render(request, "upload.html", {"form": form})