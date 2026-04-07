"""
WSGI config for bitcoinproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bitcoinproject.settings')
application = get_wsgi_application()

# Only download models if the module exists (safe for deployment)
try:
    from model_downloader import download_models
    download_models()
except ImportError:
    pass