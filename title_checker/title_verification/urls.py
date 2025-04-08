from django.urls import path
from . import views

urlpatterns = [
    path('', views.submit_title, name='submit_title'),
]