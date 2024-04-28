import pickle

footwear_lbls = []

with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\accessory_labels.pkl", "rb") as fp:
    footwear_lbls = pickle.load(fp)

print(footwear_lbls)