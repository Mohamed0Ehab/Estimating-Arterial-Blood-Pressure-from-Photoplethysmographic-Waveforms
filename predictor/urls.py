from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/sample/', views.api_sample, name='api_sample'),
]
