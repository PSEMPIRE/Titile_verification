from django.urls import path
from . import views

urlpatterns = [
    path('', views.submit_title, name='submit_title'),
    path('modify/', views.modify_and_resubmit, name='modify_and_resubmit'),
]