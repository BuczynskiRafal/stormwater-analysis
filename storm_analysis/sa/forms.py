from django import forms

from .models import SWMMModel


class SWMMModelForm(forms.ModelForm):
    class Meta:
        model = SWMMModel
        fields = ["file", "zone"]
