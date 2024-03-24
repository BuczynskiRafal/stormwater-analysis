from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.decorators import login_required

from accounts.models import User
from accounts.forms import UserForm, UserProfileForm


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("/")
    else:
        form = UserCreationForm()
    return render(request, "registration/register.html", {"form": form})


@login_required(login_url="/login")
def user_profile(request):
    user_form = UserForm(instance=request.user)
    profile_form = UserProfileForm(instance=request.user)
    password_form = PasswordChangeForm(user=request.user)

    if request.method == 'POST':
        action = request.POST.get('user_profile_forms')
        
        if action == '0':
            user_form = UserForm(request.POST, request.FILES, instance=request.user)
            if user_form.is_valid():
                user_form.save()
                messages.success(request, 'Your profile was successfully updated!')
                return redirect('accounts:user_profile')

        elif action == '1':
            password_form = PasswordChangeForm(data=request.POST, user=request.user)
            if password_form.is_valid():
                password_form.save()
                update_session_auth_hash(request, password_form.user)
                messages.success(request, 'Your password was successfully updated!')
                return redirect('accounts:user_profile')

        elif action == '2':
            profile_form = UserProfileForm(request.POST, request.FILES, instance=request.user)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, 'Your additional information was successfully updated!')
                return redirect('accounts:user_profile')
        else:
            messages.error(request, 'Invalid form submission.')
            return redirect('accounts:user_profile')
    
    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'password_form': password_form,
    }
    
    return render(request, 'accounts/user_profile.html', context)

# @login_required(login_url="/login")
# def user_profile(request):
#     user = request.user

#     try:
#         profile = user.userprofile
#     except UserProfile.DoesNotExist:
#         profile = UserProfile(user=user)
#         profile.save()

#     if request.method == "POST":
#         user_form = UserForm(request.POST, instance=user)
#         profile_form = UserProfileForm(request.POST, instance=profile)

#         if user_form.is_valid() and profile_form.is_valid():
#             user_form.save()
#             profile_form.save()
#             return redirect("accounts:user_profile")
    
#     else:
#         user_form = UserForm(instance=user)
#         profile_form = UserProfileForm(instance=profile)
    
#     context = {
#         "user_form": user_form,
#         "profile_form": user_profile
#     }
#     return render(request, "accounts/user_profile.html", context)


@login_required(login_url="/login")
def profile(request, id):
    user = User.objects.get(id=id)
    context = {"user": user}
    return render(request, "accounts/profile.html", context)


# @login_required(login_url="/login")
# def user_profile(request):
#     user = request.user
#     context = {"user": user}
#     return render(request, "accounts/user_profile.html", context)


@login_required(login_url="/login")
def update_profile(request):
    user = request.user
    user_form = UserForm(instance=user)
    profile_form = UserProfileForm(instance=UserProfile)
    if request.method == "POST":
        form = UserProfileForm(request.POST, request.FILES, instance=UserProfile)
        if form.is_valid():
            form.save()
            return redirect("profile", id=user.id)
    return render(request, "accounts/my_proifile.html", {"form": form})

# @login_required(login_url="/login")
# def update_password(request):
#     if request.method == "POST":
#         form = PasswordChangeForm(request.user, request.POST)
#         if form.is_valid():
#             user = form.save()
#             update_session_auth_hash(request, user)
#             messages.success(request, 'Your password was successfully updated!')
#             return redirect('/')
#         else:
#             messages.success(request, 'There was a error in changing password!')
#             return redirect("/update_password")
#     else:
#         form = PasswordChangeForm(request.user)
#     return render(request, "accounts/user_password.html", {"form": form})
