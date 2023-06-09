from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    return HttpResponse("hi sai")
def add(request):
    var1=request.POST["num1"]
    var2=request.POST["num2"]
    print(var1+var2)
    res=var1+var2
    return render(request,"results.html",{"result":res})
