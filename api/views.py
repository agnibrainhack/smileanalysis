# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
def post_item(request):
    if request.method == "POST":
        return render(request, "hello world")