from django import forms
from django.contrib.auth.models import User

from accounts.models import UserProfile


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'password', 'email')


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        # fields = ('phone', 'mobile', 'location', 'website_url', 'facebook_url', 'github_url', 'twitter_url')
        fields = ('website_url', 'facebook_url', 'github_url', 'twitter_url')
