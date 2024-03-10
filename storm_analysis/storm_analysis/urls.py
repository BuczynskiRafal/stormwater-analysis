from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns


urlpatterns = (
    [
        path("admin/", admin.site.urls),
        path("", include("homepage.urls")),
        path("sa/", include("sa.urls")),
        path("", include("accounts.urls"))
        # path("", include("register.urls")),
        # path("account/", include(django.contrib.auth.urls)),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    + static(settings.MEDIA_URL, documnet_root=settings.MEDIA_ROOT)
)

urlpatterns += staticfiles_urlpatterns()
