import pathlib
import textwrap
import PIL.Image
import PIL.ImageChops

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def generate_features_from_text(text):
    text += """This is a prompt entered by consumer. Give me an output which extracts the following clothing features from consumer's prompt: 
    tops features - gender, colour, type, solid/stripe, season, material
    bottoms features - gender, colour, type, season, length, material
    footwear - gender, colour, type, season, 
    accessories - gender, colour, type.
    """
    
    GOOGLE_API_KEY='AIzaSyDKWq2CnVJSs2WeVk70GrrHvuu466jnVNA'

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(text)
    return response.text

def generate_features_from_image(path):
    img = PIL.Image.open(path)
    GOOGLE_API_KEY='AIzaSyDKWq2CnVJSs2WeVk70GrrHvuu466jnVNA'

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(img)
    return generate_features_from_text(response.text)



import pickle,json, ast

with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\pickle_files\\accessory_labels.pkl", "rb") as fp:
    footwear_lbls = pickle.load(fp)

text = input("Enter prompt:")

text += f'''{footwear_lbls}
Can you provide a list of images from this which match the user's input. Act like a interactive search bar. Only return list, and no label also'''

GOOGLE_API_KEY='AIzaSyDKWq2CnVJSs2WeVk70GrrHvuu466jnVNA'

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(text)
print(type(ast.literal_eval(response.text)))
