import os
from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pickle
from .models import Footwear, Accessories, Top, Bottom
from django.core.paginator import Paginator
from urllib.parse import urlparse
from .utils import *

# Create your views here.


def index(request):
    return render(request, 'myapp/index.html')

def register(request):
    form = RegisterForm(request.POST or None)
    if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f"Welcome {username}, your account has been created!")
            return redirect('myapp:login')
    context = {
        'form': form,
    }
    return render(request, 'myapp/register.html', context)

# def image(request):
#     products = Product.objects.all()
#     print(products[0].title)
#     print(products[0].image)
#     print(type(products[0].image.url))
#     reqd_prods = Product.objects.filter(image = products[0].image)
#     print(reqd_prods)
#     return render(request, 'myapp/image.html', {'products': products})

def crazy(request):
    if request.method == "POST":
        with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\bottom_img\\bottom_labels.pkl", "rb") as fp:
            footwear_lbls = pickle.load(fp)
        for footwear_lbl in footwear_lbls:
            Bottom.objects.create(image="bottom_images/"+footwear_lbl[0], gender=footwear_lbl[1], color=footwear_lbl[2], type=footwear_lbl[3], season=footwear_lbl[4], length=footwear_lbl[5])
    return render(request, 'myapp/crazy.html')

def second(request, men_women):
    print(men_women)
    top = None
    bottom = None
    acc = None
    shoes = None
    if men_women==0:
        top = "https://img0.junaroad.com/uiproducts/19410012/zoom_1-1677344181.jpg"
        bottom = "https://i.pinimg.com/736x/20/2e/63/202e6398778c702602db8270cea799f5.jpg"
        acc = "https://images-cdn.ubuy.co.in/65631584dcb2f24d01491b9d-straw-fedora-hat-mens-fedora-hats-for.jpg"
        shoes = "https://5.imimg.com/data5/ANDROID/Default/2021/2/GP/WR/CY/123847091/product-jpeg-500x500.jpg"
    elif men_women==1:
        top = "https://assets.myntassets.com/dpr_1.5,q_60,w_400,c_limit,fl_progressive/assets/images/23201888/2023/5/15/f8d6cdb8-0f37-46bf-91ee-1d602a123acd1684164350545AthenaWhiteColourblockedBellSleeveCottonNetPeplumTop1.jpg"
        bottom = "https://assets.ajio.com/medias/sys_master/root/20240103/t5pD/65957291ddf7791519ff4bf2/-473Wx593H-466372693-brown-MODEL.jpg"
        acc = "https://5.imimg.com/data5/SELLER/Default/2023/4/297326955/XE/TP/DL/18412416/5-3-500x500.jpg"
        shoes = "https://5.imimg.com/data5/SELLER/Default/2023/12/365430631/LB/YJ/FA/6215968/casual-women-shoes.jpg"
    return render(request, 'myapp/second_page.html', {"men_women": men_women, "top": top, "bottom": bottom, "acc": acc, "shoes": shoes})

def list(request, men_women, title):
    products = []
    if title==0:
        if men_women==0:
            products = Top.objects.filter(gender="male")
            item_name = request.POST.get('item_name')
            import pickle,json, ast
            print(item_name)
            with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\accessory_labels.pkl", "rb") as fp:
                footwear_lbls = pickle.load(fp)

                

            print(products)



            if item_name != '' and item_name is not None:
                
                text = item_name
                text += f'''{footwear_lbls}
                Can you provide a list of images from this which match the user's input. Act like a interactive search bar. Only return list, and no label also. The list should be in one line, not surrounded by any brackets and each item separated by comma'''

                GOOGLE_API_KEY='AIzaSyDKWq2CnVJSs2WeVk70GrrHvuu466jnVNA'

                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-pro')

                response = model.generate_content(text)
                print(response.text)
                images = response.text.split(", ")
                images = [image.strip() for image in images]
                # print(images)
                image_urls = [f"top_images/{image}" for image in images]
                # print(image_urls)
                products = Top.objects.filter(image__in = image_urls)
            else:
                products = Top.objects.filter(gender="male")
        elif men_women==1:
            products = Top.objects.filter(gender="female")
        products = products | Top.objects.filter(gender="unisex") # Adding 2 querysets
    elif title==1:
        if men_women==0:
            products = Bottom.objects.filter(gender="male")
        elif men_women==1:
            products = Bottom.objects.filter(gender="female")
        products = products | Bottom.objects.filter(gender="unisex")
    elif title==2:
        if men_women==0:
            products = Footwear.objects.filter(gender="male")
        elif men_women==1:
            products = Footwear.objects.filter(gender="female")
        products = products | Footwear.objects.filter(gender="unisex")
    elif title==3:
        if men_women==0:
            products = Accessories.objects.filter(gender="male")
        elif men_women==1:
            products = Accessories.objects.filter(gender="female")
        products = products | Accessories.objects.filter(gender="unisex")

    
    # print(products)

    

    # pagination
    paginator = Paginator(products, 4)
    page = request.GET.get('page')
    products = paginator.get_page(page)
    return render(request, 'myapp/list.html', {"products": products, "men_women": men_women, "title": title})

def detail(request, men_women, title, id):
    print("HI")
    product = None
    table = None
    if title==0:
        product = Top.objects.get(pk=id)
        table = "Top"
    elif title==1:
        product = Bottom.objects.get(pk=id)
        table = "Bottom"
    elif title==2:
        product = Footwear.objects.get(pk=id)
        table = "Footwear"
    elif title==3:
        product = Accessories.objects.get(pk=id)
        table = "Accessories"

    recommended_bottom = None
    recommended_top = None
    recommended_acc = None
    recommended_footwear = None
    similars = None

    if table=="Top":
        li = recommend_top(product.gender.lower(), product.color.lower(), product.type.lower(), product.season.lower())
        sim_tops = [x[0] for x in li]
        sim_top_urls = [f"top_images/{sim_top}" for sim_top in sim_tops]
        print(sim_top_urls)
        sim_1 = Top.objects.filter(image = sim_top_urls[0])[0]
        sim_2 = Top.objects.filter(image = sim_top_urls[1])[0]
        sim_3 = Top.objects.filter(image = sim_top_urls[2])[0]
        similars = [sim_1, sim_2, sim_3]
        print(similars)

        query = f"I have a top wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a bottom wear that would go with it"
        best_bottom = give_best_bottom(query)
        print(best_bottom)
        recommended_bottom_url = f"bottom_images/{best_bottom[0]}"
        print(recommended_bottom_url)

        query = f"I have a top wear which is a {product.type} with color: {product.color}, for gender: {product.gender}, for season: {product.season}. Suggest me a foot wear that would go with it"
        best_footwear = give_best_footwear(query)
        print(best_footwear)
        recommended_footwear_url = f"footwear_images/{best_footwear[0]}"

        query = f"I have a top wear which is a {product.type} with color: {product.color}, for gender: {product.gender}, for season: {product.season}. Suggest me an accessory that would go with it"
        best_acc = give_best_accessory(query)
        print(best_acc)
        recommended_acc_url = f"acc_images/{best_acc[0]}"
        print(recommended_acc_url)

        recommended_bottom = Bottom.objects.filter(image = recommended_bottom_url)[0]
        recommended_footwear = Footwear.objects.filter(image = recommended_footwear_url)[0]
        recommended_acc = Accessories.objects.filter(image = recommended_acc_url)[0]
        print(recommended_acc)

    elif table=="Bottom":
        li = recommend_bottom(product.gender.lower(), product.color.lower(), product.type.lower(), product.season.lower(), product.length.lower())
        sim_tops = [x[0] for x in li]
        sim_top_urls = [f"bottom_images/{sim_top}" for sim_top in sim_tops]
        print(sim_top_urls)
        sim_1 = Bottom.objects.filter(image = sim_top_urls[0])[0]
        sim_2 = Bottom.objects.filter(image = sim_top_urls[1])[0]
        sim_3 = Bottom.objects.filter(image = sim_top_urls[2])[0]
        similars = [sim_1, sim_2, sim_3]
        print(similars)

        query = f"I have a bottom wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}, of length: {product.length.lower()}. Suggest me a top wear that would go with it"
        best_top = give_best_top(query)
        print(best_top)
        recommended_top_url = f"top_images/{best_top[0]}"
        print(recommended_top_url)

        query = f"I have a bottom wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}, of length: {product.length.lower()}. Suggest me a foot wear that would go with it"
        best_footwear = give_best_footwear(query)
        print(best_footwear)
        recommended_footwear_url = f"footwear_images/{best_footwear[0]}"

        query = f"I have a bottom wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}, of length: {product.length.lower()}. Suggest me an accessory that would go with it"
        best_acc = give_best_accessory(query)
        print(best_acc)
        recommended_acc_url = f"acc_images/{best_acc[0]}"

        recommended_top = Top.objects.filter(image = recommended_top_url)[0]
        print(recommended_top)
        recommended_footwear = Footwear.objects.filter(image = recommended_footwear_url)[0]
        recommended_acc = Accessories.objects.filter(image = recommended_acc_url)[0]

    elif table=="Footwear":
        li = recommend_footwear(product.gender.lower(), product.color.lower(), product.type.lower(), product.season.lower())
        sim_tops = [x[0] for x in li]
        sim_top_urls = [f"footwear_images/{sim_top}" for sim_top in sim_tops]
        print(sim_top_urls)
        sim_1 = Footwear.objects.filter(image = sim_top_urls[0])[0]
        sim_2 = Footwear.objects.filter(image = sim_top_urls[1])[0]
        sim_3 = Footwear.objects.filter(image = sim_top_urls[2])[0]
        similars = [sim_1, sim_2, sim_3]
        print(similars)

        query = f"I have a foot wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a top wear that would go with it"
        best_top = give_best_top(query)
        print(best_top)
        recommended_top_url = f"top_images/{best_top[0]}"

        query = f"I have a foot wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a bottom wear that would go with it"
        best_bottom = give_best_bottom(query)
        print(best_bottom)
        recommended_bottom_url = f"bottom_images/{best_bottom[0]}"

        query = f"I have a foot wear which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me an accessory that would go with it"
        best_acc = give_best_accessory(query)
        print(best_acc)
        recommended_acc_url = f"acc_images/{best_acc[0]}"

        recommended_top = Top.objects.filter(image = recommended_top_url)[0]
        print(recommended_top)
        recommended_bottom = Bottom.objects.filter(image = recommended_bottom_url)[0]
        recommended_acc = Accessories.objects.filter(image = recommended_acc_url)[0]

    elif table=="Accessories":
        li = recommend_accessory(product.gender.lower(), product.color.lower(), product.type.lower(), product.season.lower())
        sim_tops = [x[0] for x in li]
        sim_top_urls = [f"acc_images/{sim_top}" for sim_top in sim_tops]
        print(sim_top_urls)
        sim_1 = Accessories.objects.filter(image = sim_top_urls[0])[0]
        sim_2 = Accessories.objects.filter(image = sim_top_urls[1])[0]
        sim_3 = Accessories.objects.filter(image = sim_top_urls[2])[0]
        similars = [sim_1, sim_2, sim_3]
        print(similars)

        query = f"I have an accessory which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a top wear that would go with it"
        best_top = give_best_top(query)
        print(best_top)
        recommended_top_url = f"top_images/{best_top[0]}"

        query = f"I have an accessory which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a bottom wear that would go with it"
        best_bottom = give_best_bottom(query)
        print(best_bottom)
        recommended_bottom_url = f"bottom_images/{best_bottom[0]}"

        query = f"I have an accessory which is a {product.type.lower()} with color: {product.color.lower()}, for gender: {product.gender.lower()}, for season: {product.season.lower()}. Suggest me a foot wear that would go with it"
        best_footwear = give_best_footwear(query)
        print(best_footwear)
        recommended_footwear_url = f"footwear_images/{best_footwear[0]}"

        recommended_top = Top.objects.filter(image = recommended_top_url)[0]
        print(recommended_top)
        recommended_bottom = Bottom.objects.filter(image = recommended_bottom_url)[0]
        recommended_footwear = Footwear.objects.filter(image = recommended_footwear_url)[0]



    return render(request, 'myapp/detail.html', {"product": product, "men_women": men_women, "title": title, "recommended_bottom": recommended_bottom, "recommended_footwear": recommended_footwear, "recommended_acc": recommended_acc, "recommended_top": recommended_top, "similars": similars})
        