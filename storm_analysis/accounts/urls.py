from django.urls import path
from django.contrib.auth import views as auth_views
from accounts import views

app_name = "accounts"

urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html", next_page="/"), name="login"),
    path('logout/', auth_views.LogoutView.as_view(next_page="/"), name='logout'),
    path("register/", views.register, name="register"),
    path("myprofile/", views.my_profile, name="myprofile"),
    path("profile/<str:id>", views.profile, name="profile"),
    path("update_profile/", views.update_profile, name="update_profile"),
    path("update_password/", views.update_password, name="update_password"),
]