from django.db import models


class UserSWMMModel(models.Model):
    file = models.FileField(upload_to="user_models/")
