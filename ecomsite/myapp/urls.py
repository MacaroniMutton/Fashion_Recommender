from django.contrib import admin
from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views

app_name = 'myapp'
urlpatterns = [
    path("", views.index, name="index"),
    # path("image/", views.image, name="image"),
    path('login/', auth_views.LoginView.as_view(template_name='myapp/login.html'), name="login"),
    path('logout/', auth_views.LogoutView.as_view(template_name='myapp/logout.html'), name="logout"),
    path("register/", views.register, name="register"),
    path("crazy", views.crazy, name="crazy"),
    path("second/<int:men_women>/", views.second, name="second"),
    path("list/<int:men_women>/<int:title>", views.list, name="list"),
    path("detail/<int:men_women>/<int:title>/<int:id>", views.detail, name="detail"),
]
