"""
Django settings module with environment-based configuration.

Usage:
    Set DJANGO_ENV environment variable to select configuration:
    - development (default)
    - production
    - testing

Example:
    export DJANGO_ENV=production
    python manage.py runserver
"""

import os

env = os.getenv("DJANGO_ENV", "development")

if env == "production":
    from .production import *  # noqa: F401, F403
elif env == "testing":
    from .testing import *  # noqa: F401, F403
else:
    from .development import *  # noqa: F401, F403
