from django.urls import path
from . import views
urlpatterns=[
    path("ipl/",views.ipl,name="ipl"),
    path("",views.home,name="home"),
    path("login/",views.login,name="login"),
    path("register/",views.register,name="register"),
    path("logout/",views.logout,name="logout"),
    path("test/",views.test,name="test")
]