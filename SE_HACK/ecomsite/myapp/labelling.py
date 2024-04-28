import os
import pickle


print(os.listdir('C:\\Users\\Manan Kher\\OneDrive\\Documents\\bottom_img'))

images = os.listdir('C:\\Users\\Manan Kher\\OneDrive\\Documents\\bottom_img')
images = sorted(images, key=lambda x: int(x.split(".")[0]))

print(images)

labels = []
print()
# gender, color, type, season, length
labels.append([images[0], "female", "black", "pants", "all", "full"])
labels.append([images[1], "male", "beige", "pants", "winter", "full"])
labels.append([images[2], "unisex", "red", "pants", "winter", "full"])
labels.append([images[3], "male", "blue", "jeans", "all", "full"])
labels.append([images[4], "male", "black", "jeans", "all", "full"])
labels.append([images[5], "female", "blue", "jeans", "all", "full"])
labels.append([images[6], "female", "brown", "cargo", "all", "full"])
labels.append([images[7], "unisex", "beige", "cargo", "all", "full"])
labels.append([images[8], "unisex", "green", "cargo", "all", "full"])
labels.append([images[9], "female", "blue", "shorts", "summer", "short"])
labels.append([images[10], "male", "black", "shorts", "summer", "short"])
labels.append([images[11], "female", "blue", "shorts", "summer", "short"])
labels.append([images[12], "female", "pink", "skirt", "summer", "short"])
labels.append([images[13], "female", "white", "skirt", "winter", "full"])
labels.append([images[14], "female", "blue", "skirt", "summer", "short"])
labels.append([images[15], "female", "grey", "leggings", "all", "full"])
labels.append([images[16], "female", "white", "leggings", "all", "full"])
labels.append([images[17], "female", "orange", "leggings", "all", "full"])
labels.append([images[18], "unisex", "beige", "joggers", "summer", "full"])
labels.append([images[19], "female", "pink", "joggers", "summer", "full"])
labels.append([images[20], "unisex", "green", "joggers", "summer", "full"])
print(labels)


with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\bottom_img\\bottom_labels.pkl', 'wb') as fp:
    pickle.dump(labels, fp)

# print()
# print()


# with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\acc_img\\labels.pkl', 'rb') as fp:
#     li_1 = pickle.load(fp)


# with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\acc_img\\acc_labels.pkl', 'rb') as fp:
#     li_2 = pickle.load(fp)

# li_final = li_1 + li_2
# print(li_final)

# with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\acc_img\\accessory_labels.pkl', 'wb') as fp:
#     pickle.dump(li_final, fp)
