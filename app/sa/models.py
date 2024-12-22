from accounts.models import User
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


def validate_file_extension(file):
    if not file.name.endswith(".inp"):
        raise ValidationError(_("Invalid file extension. Required extension is .inp"))


class SWMMModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to="user_models/", validators=[validate_file_extension])
    ZONE_CHOICES = (
        (1, "Zone 1"),
        (2, "Zone 2"),
        (3, "Zone 3"),
        (4, "Zone 4"),
    )
    zone = models.IntegerField(choices=ZONE_CHOICES, null=False, blank=False)
