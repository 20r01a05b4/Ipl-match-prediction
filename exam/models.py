from django.db import models

class Destination(models.Model):
   
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    para = models.TextField()
    img = models.ImageField(upload_to='images')


    # Other fields...
