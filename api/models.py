# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


# Create your models here.
class Note(models.Model):
    
    image = models.TextField()
    time = models.TextField(default='1234567')

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image
