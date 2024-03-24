from django import forms
from accounts.models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["email", "phone"]

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["bio", "website_url", "facebook_url", "github_url", "twitter_url"]
