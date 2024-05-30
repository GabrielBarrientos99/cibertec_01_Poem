from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('poem/<int:poem_id>/', views.poem_detail, name='poem_detail'),
    path('profile/', views.user_profile, name='user_profile'),
    path('save_poem/', views.save_poem, name='save_poem'),
     path('saved_poems/', views.saved_poems, name='saved_poems'),
    path('analyze_poem/', views.analyze_poem, name='analyze_poem'),
    path('delete_poem/<int:poem_id>/', views.delete_poem, name='delete_poem'),
   

]
