from django.urls import path , include
from . import views

urlpatterns = [
    path("",              views.index,          name="index"),
    path("forecast/",     views.forecast_view,  name="forecast"),
    path("dashboard/",    views.dashboard_view, name="dashboard"),
    path("live/",         views.live_dashboard, name="live"),
    path("api/forecast/", views.api_forecast,   name="api_forecast"),
]