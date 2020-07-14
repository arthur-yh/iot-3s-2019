from django.urls import path
from . import views

app_name = 'dianbiao'
urlpatterns = [
    path('', views.get_current_date, name='home'),
    path('show/', views.index, name='index'),
    path('control/', views.control, name='control'),
    path('show/refresh/', views.refresh, name='refresh'),
    path('alarm/', views.alarm, name='alarm'),
    path('alarm/refresh_alarm/', views.refresh_alarm, name='refresh_alarm'),
]
