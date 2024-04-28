


GOOGLE_API_KEY='AIzaSyDKWq2CnVJSs2WeVk70GrrHvuu466jnVNA'


import google.generativeai as genai



def nearest_neighbour(item,labels):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # Create the model, looking for the single closest neighbor
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(labels)

    # Find the closest vector
    distances, indices = nn.kneighbors([item])

    # print("Closest vector is at index:", indices[0][0], "with distance:", distances[0][0])
    return indices[0][0]



def give_best_top(query):
    colours = ["black", "white", "blue", "grey", "yellow"]
    types = ["tshirt","shirt","sweater","jacket","crop top","dress"]
    seasons = ["all","summer","winter"]

    query+=f"Make sure all The features like colour should belong to list {colours},gender should be male, female or unisex, type of shirt should belong to list {types}, and season should belong to list {seasons} and specify accordingly please in this format 'gender colour type season', give only these values and give only one option"
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    return find_best_match_top(response.text)


def encode_top(item):
    color_encoding = {"black": 0, "white": 1, "blue": 2, "grey": 3, "yellow": 4, "other": 5}

    type_encoding = {"tshirt": 0, "shirt": 1, "sweater": 2, "jacket": 3, "crop-top": 4, "dress": 5, "other": 6}

    season_encoding = {"all": 0, "winter": 3, "summer": 7, "other": 2} #changing difference between values from 1 to add more weightage perhaps
    
    gender_encoding = {"unisex" : 0, "male" : 1, "female" : 2}
    
    # Extract data from each item
    gender, color, type, season = item

    # Encode color, type, and season
    encoded_gender = gender_encoding.get(gender)
    encoded_color = color_encoding.get(color, color_encoding["other"])  # Handle missing values
    encoded_type = type_encoding.get(type, type_encoding["other"])
    encoded_season = season_encoding.get(season, season_encoding["other"])

    # Create a new list with encoded values
    encoded_item = [encoded_gender, encoded_color, encoded_type, encoded_season]

    return encoded_item

def find_best_match_top(item):
    top_labels = [
    ["1.jpg","male","black","tshirt","all"],
    ["2.jpg","unisex","white","tshirt","all"],
    ["3.png","male","blue","tshirt","all"],
    ["4.jpg","male","blue","shirt","all"],
    ["5.jpg","male","white","shirt","all"],
    ["6.png","male","black","shirt","all"],
    ["7.jpg","unisex","white","sweater","all"],
    ["8.jpg","unisex","black","sweater","winter"],
    ["9.jpg","unisex","blue","sweater","winter"],
    ["10.jpg","unisex","grey","sweater","all"],
    ["11.jpg","unisex","black","jacket","winter"],
    ["12.jpg","female","blue","jacket","winter"],
    ["13.jpg","female","white","crop-top","summer"],
    ["14.jpg","female","blue","crop-top","summer"],
    ["15.jpg","female","grey","crop-top","summer"],
    ["16.jpg","female","black","dress","summer"],
    ["17.jpg","female","blue","dress","summer"],
    ["18.jpg","female","yellow","dress","summer"],
    ]
    encoded_labels = []
    item = item.lower().split()
    for label in top_labels:
        encoded_labels.append(encode_top(label[1:]))
    encoded_item = encode_top(item)
    
    index = nearest_neighbour(encoded_item,encoded_labels)
    
    return top_labels[index]



def give_best_footwear(query):
    colours = ["black","green","brown","white","blue","red","light-brown","pink"]
    seasons = ["summer","winter","all"]
    types = ["boots","crocs","heels","shoes","sandals","sneakers","slippers"]
    query+=f"Make sure all The features like colour should belong to list {colours},gender should be male, female or unisex, type of shirt should belong to list {types}, and season should belong to list {seasons} and specify accordingly please in this format 'gender colour type season', give only these values dont mention the labels and give only one option"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    print(response.text)
    return find_best_match_footwear(response.text)

def encode_footwear(item):
    color_encoding = {"black": 0, "white": 1, "blue": 2, "red": 3, "green": 4, "brown": 5, "light-brown": 6, "pink": 7, "other": 8}

    season_encoding = {"summer": 7, "winter": 3, "all": 0, "other": 2}

    type_encoding = {"boots": 0, "crocs": 1, "heels": 2, "shoes": 3, "sandals": 4, "sneakers": 5, "slippers": 6, "other": 7}
    
    gender_encoding = {"unisex" : 0, "male" : 1, "female" : 2}
    
    gender, color, type, season = item

    # Encode color, type, and season
    encoded_gender = gender_encoding.get(gender)
    encoded_color = color_encoding.get(color, color_encoding["other"])  # Handle missing values
    encoded_type = type_encoding.get(type, type_encoding["other"])
    encoded_season = season_encoding.get(season, season_encoding["other"])

    # Create a new list with encoded values
    encoded_item = [encoded_gender, encoded_color, encoded_type, encoded_season]

    return encoded_item

def find_best_match_footwear(item):
    import pickle

    footwear_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\footwear_labels.pkl", 'rb') as fp:
        footwear_labels = pickle.load(fp)
    for label in footwear_labels:
        if label[2] == "light brown":
            label[2] = "light-brown"
    
    item = item.lower().split()
    
    encoded_labels = []
    
    for label in footwear_labels:
        encoded_labels.append(encode_footwear(label[1:]))
    
    encoded_item = encode_footwear(item)
    
    index = nearest_neighbour(encoded_item,encoded_labels)
    
    return footwear_labels[index]

def find_best_match_accessory(item):
    import pickle

    accessory_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\accessory_labels.pkl", 'rb') as fp:
        accessory_labels = pickle.load(fp)
    for label in accessory_labels:
        if label[2] == "rose gold":
            label[2] = "rose-gold"
    
    item = item.lower().split()
    
    encoded_labels = []
    
    for label in accessory_labels:
        encoded_labels.append(encode_accessory(label[1:]))
    
    encoded_item = encode_accessory(item)
    
    index = nearest_neighbour(encoded_item,encoded_labels)
    
    return accessory_labels[index]


def give_best_accessory(query):
    colours = ["white","brown","black","blue","green","silver","gold","rose-gold","pink"]
    types = ["hat","cap","sunglasses","earrings","necklace","handbag"]
    seasons = ["winter","summer","all"]
    
    query+=f"Make sure all The features like colour should belong to list {colours},gender should be male, female or unisex, type of shirt should belong to list {types}, and season should belong to list {seasons} and specify accordingly please in this format 'gender colour type season', give only these values and give only one option"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    print(response.text)
    return find_best_match_accessory(response.text)

def encode_accessory(item):
    color_encoding = { "white": 0, "brown": 1, "black": 2, "blue": 3, "green": 4, "silver": 5, "gold": 6, "rose-gold": 7, "pink": 8, "other": 9  }
    
    gender_encoding = {"unisex" : 0, "male" : 1, "female" : 2}

    type_encoding = {"hat": 0,"cap": 1,"sunglasses": 2,"earrings": 3,"necklace": 4,"handbag": 5,"other": 6}

    season_encoding = {"winter": 3,"summer": 7,"all": 0,"other": 2}
    
    gender, color, type, season = item

    # Encode color, type, and season
    encoded_gender = gender_encoding.get(gender)
    encoded_color = color_encoding.get(color, color_encoding["other"])  # Handle missing values
    encoded_type = type_encoding.get(type, type_encoding["other"])
    encoded_season = season_encoding.get(season, season_encoding["other"])

    # Create a new list with encoded values
    encoded_item = [encoded_gender, encoded_color, encoded_type, encoded_season]

    return encoded_item

def find_best_match_bottom(item):
    import pickle

    bottom_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\bottom_labels.pkl", 'rb') as fp:
        bottom_labels = pickle.load(fp)
        
    item = item.lower().split()
    
    encoded_labels = []
    
    for label in bottom_labels:
        encoded_labels.append(encode_bottom(label[1:]))
    
    encoded_item = encode_bottom(item)
    
    index = nearest_neighbour(encoded_item,encoded_labels)
    
    return bottom_labels[index]


def encode_bottom(item):
    color_encoding = { "black": 0, "beige": 1, "red": 2, "blue": 3, "brown": 4, "green": 5, "pink": 6, "white": 7, "grey": 8, "orange": 9, "other": 10}

    type_encoding = { "pants": 0, "jeans": 1, "cargo": 2, "shorts": 3, "skirt": 4, "leggings": 5, "joggers": 6, "other": 7 }

    season_encoding = {"all": 0,"winter": 3,"summer": 1,"other": 2}
    
    length_encoding = {"full" : 0, "short" : 2, "other" : 4}
    
    gender_encoding = {"unisex" : 0, "male" : 1, "female" : 2}
    
    gender, color, type, season, length = item

    # Encode color, type, and season
    encoded_gender = gender_encoding.get(gender)
    encoded_color = color_encoding.get(color, color_encoding["other"])  # Handle missing values
    encoded_type = type_encoding.get(type, type_encoding["other"])
    encoded_season = season_encoding.get(season, season_encoding["other"])
    encoded_length = length_encoding.get(length, length_encoding["other"])

    # Create a new list with encoded values
    encoded_item = [encoded_gender, encoded_color, encoded_type, encoded_season, encoded_length]

    return encoded_item


def give_best_bottom(query)      :
    colours = ["black","beige","red","blue","brown","green","pink","white","grey","orange"]
    types = ["pants","jeans","cargo","shorts","skirt","leggings","joggers"]
    seasons = ["all","winter","summer"]
    length = ["full","short"]
    
    query+=f"Make sure all The features like colour should belong to list {colours},gender should be male, female or unisex, type of shirt should belong to list {types}, and season should belong to list {seasons} and length should be either full or short, specify accordingly please in this format 'gender colour type season length', give only these values and give only one option"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    print(response.text)
    return find_best_match_bottom(response.text)


# import pickle

# bottom_labels = []

# with open('C:\\Users\\krish\\OneDrive\\Desktop\\SE Hackathon\\bottom_labels.pkl', 'rb') as fp:
#     bottom_labels = pickle.load(fp)

# print(bottom_labels)

# print(labels[find_best_match_top(["male","grey","shirt","all"])])
# prompt = input("Enter prompt: ")

# print(give_best_bottom(prompt))

# suggestion = give_best_top(f"{prompt}").lower().split()

# print(suggestion)

# recommendation = labels[find_best_match_top(suggestion)]

# print(recommendation)



def recommend_neighbours(item,labels):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np


    nn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
    nn.fit(labels)

    # Find the closest vector
    distances, indices = nn.kneighbors([item])

    # print("Closest vector is at index:", indices[0][0], "with distance:", distances[0][0])
    return indices[0]

def recommend_top(gender,colour,type,season):
    item = [gender.lower(),colour.lower(),type.lower(),season.lower()]
    encoded_item = encode_top(item)
    top_labels = [
    ["1.jpg","male","black","tshirt","all"],
    ["2.jpg","unisex","white","tshirt","all"],
    ["3.png","male","blue","tshirt","all"],
    ["4.jpg","male","blue","shirt","all"],
    ["5.jpg","male","white","shirt","all"],
    ["6.png","male","black","shirt","all"],
    ["7.jpg","unisex","white","sweater","all"],
    ["8.jpg","unisex","black","sweater","winter"],
    ["9.jpg","unisex","blue","sweater","winter"],
    ["10.jpg","unisex","grey","sweater","all"],
    ["11.jpg","unisex","black","jacket","winter"],
    ["12.jpg","female","blue","jacket","winter"],
    ["13.jpg","female","white","crop-top","summer"],
    ["14.jpg","female","blue","crop-top","summer"],
    ["15.jpg","female","grey","crop-top","summer"],
    ["16.jpg","female","black","dress","summer"],
    ["17.jpg","female","blue","dress","summer"],
    ["18.jpg","female","yellow","dress","summer"],
    ]
    encoded_labels = []
    for label in top_labels:
        encoded_labels.append(encode_top(label[1:]))

    indices = recommend_neighbours(encoded_item,encoded_labels)[1:]
    recommendations = []
    for i in indices:
        recommendations.append(top_labels[i])
    return recommendations

def recommend_bottom(gender,colour,type,season,length):
    item = [gender.lower(),colour.lower(),type.lower(),season.lower(),length.lower()]
    encoded_item = encode_bottom(item)
    import pickle

    bottom_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\bottom_labels.pkl", 'rb') as fp:
        bottom_labels = pickle.load(fp)
    encoded_labels = []

    for label in bottom_labels:
        encoded_labels.append(encode_bottom(label[1:]))

    indices = recommend_neighbours(encoded_item,encoded_labels)[1:]
    recommendations = []
    for i in indices:
        recommendations.append(bottom_labels[i])
    return recommendations

def recommend_footwear(gender,colour,type,season):
    item = [gender.lower(),colour.lower(),type.lower(),season.lower()]
    encoded_item = encode_footwear(item)
    import pickle

    footwear_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\footwear_labels.pkl", 'rb') as fp:
        footwear_labels = pickle.load(fp)
    encoded_labels = []

    for label in footwear_labels:
        encoded_labels.append(encode_footwear(label[1:]))

    indices = recommend_neighbours(encoded_item,encoded_labels)[1:]
    recommendations = []
    for i in indices:
        recommendations.append(footwear_labels[i])
    return recommendations

def recommend_accessory(gender,colour,type,season):
    item = [gender.lower(),colour.lower(),type.lower(),season.lower()]
    encoded_item = encode_accessory(item)
    import pickle

    accessory_labels = []

    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\accessory_labels.pkl", 'rb') as fp:
        accessory_labels = pickle.load(fp)
    encoded_labels = []

    for label in accessory_labels:
        encoded_labels.append(encode_accessory(label[1:]))

    indices = recommend_neighbours(encoded_item,encoded_labels)[1:]
    recommendations = []
    for i in indices:
        recommendations.append(accessory_labels[i])
    return recommendations