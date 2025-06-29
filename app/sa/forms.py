from django import forms

from .models import SWMMModel


class SWMMModelForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Update zone field to show frost zone values in labels
        zone_choices = []
        for zone_num, frost_value in SWMMModel.ZONE_CHOICES:
            zone_choices.append((zone_num, f"Zone {zone_num} (Frost depth: {frost_value}m)"))
        self.fields["zone"].choices = zone_choices
        self.fields["zone"].widget.attrs.update({"class": "form-select"})
        self.fields["file"].widget.attrs.update({"class": "form-control", "accept": ".inp"})

    class Meta:
        model = SWMMModel
        fields = ["file", "zone"]
        labels = {"file": "SWMM Model File (.inp)", "zone": "Frost Zone"}
        help_texts = {
            "file": "Upload your SWMM model file with .inp extension",
            "zone": "Select the appropriate frost zone for your region",
        }
