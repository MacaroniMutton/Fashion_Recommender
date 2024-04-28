from django.contrib import admin
from .models import Footwear, Accessories, Top, Bottom, Purchase_History_Bottom, Purchase_History_Top, Purchase_History_Accessories, Purchase_History_Footwear

# Register your models here.


admin.site.register(Footwear)
admin.site.register(Accessories)
admin.site.register(Top)
admin.site.register(Bottom)
admin.site.register(Purchase_History_Bottom)
admin.site.register(Purchase_History_Top)
admin.site.register(Purchase_History_Accessories)
admin.site.register(Purchase_History_Footwear)