from __future__ import unicode_literals

from django.http import HttpResponse, JsonResponse
from tastypie.resources	 import ModelResource
from api.models import Note
from tastypie.authorization import Authorization
import api.image_check as im

from django.db.models import CharField
from django.db.models.functions import Cast


class NoteResource(ModelResource):
    class Meta:
        index_exclude_fields = ['image']
        queryset = Note.objects.all()
        always_return_data = True
        
        authorization = Authorization()

    def dehydrate(self, bundle):

        if bundle.request.method == 'POST':
            value = Note.objects.last()
            val = str(value)
            obj = im.Image(val)
            bundle.data['image_result'] = obj.ret()   #obj.func(val)
            bundle.data['image'] = " "

        return bundle
