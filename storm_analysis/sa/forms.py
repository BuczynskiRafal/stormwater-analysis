from django import forms

from .models import UserSWMMModel


class UserSWMMModelForm(forms.ModelForm):
    class Meta:
        model = UserSWMMModel
        fields = "__all__"
