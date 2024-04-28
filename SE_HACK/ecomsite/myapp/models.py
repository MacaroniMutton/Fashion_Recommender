from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Footwear(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="footwear_images")
    gender = models.CharField(max_length=10, default="female")
    color = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    season = models.CharField(max_length=100)
    price = models.FloatField(default=50)

class Accessories(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="acc_images")
    gender = models.CharField(max_length=10, default="female")
    color = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    season = models.CharField(max_length=100)
    price = models.FloatField(default=50)

class Top(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="top_images")
    gender = models.CharField(max_length=10, default="female")
    color = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    season = models.CharField(max_length=100)
    price = models.FloatField(default=50)

class Bottom(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="bottom_images")
    gender = models.CharField(max_length=10, default="female")
    color = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    season = models.CharField(max_length=100)
    length = models.CharField(max_length=100)
    price = models.FloatField(default=50)

class Purchase_History_Footwear(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    footwear = models.ForeignKey(Footwear, on_delete=models.CASCADE)

class Purchase_History_Accessories(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    acc = models.ForeignKey(Accessories, on_delete=models.CASCADE)

class Purchase_History_Top(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    top = models.ForeignKey(Top, on_delete=models.CASCADE)

class Purchase_History_Bottom(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    bottom = models.ForeignKey(Bottom, on_delete=models.CASCADE)

