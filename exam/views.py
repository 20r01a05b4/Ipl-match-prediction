from django.shortcuts import render
from django.http import HttpResponse
from  .models import Destination



def exam(request):
  
    dest = Destination.objects.all()
    
    return render(request, "home1.html", {"dest1": dest})

