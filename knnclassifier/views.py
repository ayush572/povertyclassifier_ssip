from django.shortcuts import render

# Create your views here.
def index(req):
    context={'a':1}
    return render(req, "index.html", context)
