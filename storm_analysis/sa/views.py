from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .forms import UserSWMMModelForm


@login_required
def analysis(request):
    return render(request, "sa/analysis.html")

@login_required
def history(request):
    return render(request, "sa/history.html")

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
