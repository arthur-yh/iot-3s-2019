from django.shortcuts import render

from django.http import HttpResponse

def index(request):
    return HttpResponse("Start our dianbiao-web application!")
# Create your views here.
