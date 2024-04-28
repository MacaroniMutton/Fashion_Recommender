#!/usr/bin/env python
# coding: utf-8

# In[17]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[18]:


# df = pd.read_csv("/kaggle/input/training-articles/train_articles.csv", index_col=False)


# In[19]:


# columns_to_drop = ["Unnamed: 0","product_code", "product_type_no","graphical_appearance_no","colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id", "department_no", "index_code", "index_group_no", "section_no", "garment_group_no"]


# In[20]:


# df = df.drop(columns=columns_to_drop)


# In[21]:


# df[:1]


# In[22]:


# def extract_article_no(name):
#     return name[1:]


# In[23]:


# image_dir = "/kaggle/input/h-and-m-personalized-fashion-recommendations/images"

# image_article_no = []

# for subdir, dirs, files in os.walk(image_dir):
#     for file in files:
#         file_path = os.path.join(subdir, file)
#         image_article_no.append(extract_article_no(file_path))


# In[24]:


# image_paths = image_article_no[:1000]


# In[25]:


# def check_match(value):
#     for path in image_paths:
#         if str(value) == (path.split('/')[-1])[1:10]:
#             return path
#     return None
# check_match(570177001)
# # Add a new column 'label' to the DataFrame and populate it with matched values
# df['label'] = df['article_id'].apply(check_match)


# In[28]:


# df_test = df.copy()


# In[29]:


# df_test.to_csv('/kaggle/working/df_test.csv', index=False)


# In[30]:


# df_test = pd.read_csv('/kaggle/working/df_test.csv')


# In[31]:


# df_test.head()


# In[32]:


# df_test = df_test.dropna()


# In[33]:


# df_test[:5]


# In[34]:


# df_test["description"] = df_test["perceived_colour_master_name"] + ' ' + df_test["graphical_appearance_name"] + ' ' + df_test["product_group_name"] + ' ' + df_test["product_type_name"] + ' ' + df_test["index_group_name"]


# In[35]:


# df_test[:5]


# In[36]:


# del_col = ["product_group_name", "graphical_appearance_name","colour_group_name",	"perceived_colour_value_name",	"perceived_colour_master_name", "department_name",	"index_name",	"index_group_name",	"section_name",	"garment_group_name"]
# # df_desc = df_test.drop(columns=del_col)


# In[37]:


# values_to_delete = ["Ballerinas",                    
# "Cap/peaked"  ,             
# "Outdoor trousers"  ,       
# # "Underwear body ",           
# "Heeled sandals"        ,   
# "Pyjama jumpsuit/playsuit",
# "Bodysuit"   ,         
# "Kids Underwear top" ,     
# "Sunglasses"  ,              
# "Hat/brim"  ,               
# "Outdoor Waistcoat" ,        
# "Outdoor overall",           
# "Ring"  ,                    
# "Waterbottle" ,               
# "Night gown",              
# "Bikini top",               
# "Hair/alice band",
# "Pyjama set",            
# "Belt",                   
# "Necklace",                
# "Robe",                    
# "Swimwear set", "Pyjama bottom",
# "Dungarees",                    
# "Earring",                      
# "Underwear Tights", "Hair string", 
# "Other accessories",           
# "Scarf",                        
# "Other shoe", "Underwear body",
# "Swimsuit"  ,
# "Swimwear bottom",
# "Jumpsuit/Playsuit", "Garment Set", "Underwear bottom"]
# df_desc = df_desc[~df_desc.isin(values_to_delete).any(axis=1)]


# In[38]:


# df_desc["product_type_name"].value_counts()


# # Splitting into train and test set and removing sampling bias

# In[39]:


# from sklearn.model_selection import train_test_split


# # Splitting the DataFrame into training and testing sets
# train_df, test_df = train_test_split(df_desc, test_size=0.2, random_state=42, stratify=df_desc['product_type_name'])

# # Display the sizes of the resulting DataFrames
# print("Train set size:", len(train_df))
# print("Test set size:", len(test_df))


# In[40]:


# train_df_image_text = train_df.drop(["article_id",	"prod_name",	"product_type_name",	"detail_desc"], axis=1)
# test_df_image_text = test_df.drop(["article_id",	"prod_name",	"product_type_name",	"detail_desc"], axis=1)


# In[41]:


# train_df_image_text.to_csv("/kaggle/working/train_df_image_text.csv")


# In[42]:


# test_df_image_text.to_csv("/kaggle/working/test_df_image_text.csv")


# # Now trying Image Captions

# In[43]:


get_ipython().system('pip -q install kaggle')


# In[44]:


import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm


# In[45]:


# Train
captions = pd.read_csv('/kaggle/working/train_df_image_text.csv', index_col=None)
captions['label'] = captions['label'].apply(
    lambda x: f'/{x}')
captions.drop(["Unnamed: 0"], axis=1, inplace=True)
# captions.head()

# Test
captions_test = pd.read_csv('/kaggle/working/test_df_image_text.csv', index_col=None)
captions_test['label'] = captions_test['label'].apply(
    lambda x: f'/{x}')
captions_test.drop(["Unnamed: 0"], axis=1, inplace=True)
# captions_test.head()


# In[46]:


# Preprocessing text / caption
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text


# In[47]:


captions['description'] = captions['description'].apply(preprocess)
captions_test['description'] = captions_test['description'].apply(preprocess)


# In[48]:


captions_test.head()


# In[49]:


# Only run once
random_row = captions.sample(1).iloc[0]
print(random_row.description)
print()
im = Image.open(random_row.label)
im


# In[74]:


MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 256


# In[75]:


tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH)

tokenizer.adapt(captions['description'])


# In[76]:


word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)


# In[77]:


# img_to_cap_vector = collections.defaultdict(list)
# for img, cap in zip(captions['label'], captions['description']):
#     img_to_cap_vector[img].append(cap)

# img_keys = list(img_to_cap_vector.keys())
# random.shuffle(img_keys)

# slice_index = int(len(img_keys)*0.8)
# img_name_train_keys, img_name_val_keys = (img_keys[:slice_index], 
#                                           img_keys[slice_index:])

# train_imgs = []
# train_captions = []
# for imgt in img_name_train_keys:
#     capt_len = len(img_to_cap_vector[imgt])
#     train_imgs.extend([imgt] * capt_len)
#     train_captions.extend(img_to_cap_vector[imgt])

# val_imgs = []
# val_captions = []
# for imgv in img_name_val_keys:
#     capv_len = len(img_to_cap_vector[imgv])
#     val_imgs.extend([imgv] * capv_len)
#     val_captions.extend(img_to_cap_vector[imgv])


# In[78]:


img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(captions['label'], captions['description']):
    img_to_cap_vector[img].append(cap)
    
img_to_cap_vector_test = collections.defaultdict(list)
for img, cap in zip(captions_test['label'], captions_test['description']):
    img_to_cap_vector_test[img].append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)
img_keys_test = list(img_to_cap_vector_test.keys())
random.shuffle(img_keys_test)

img_name_train_keys = img_keys
img_name_test_keys =  img_keys_test

train_imgs = []
train_captions = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    train_imgs.extend([imgt] * capt_len)
    train_captions.extend(img_to_cap_vector[imgt])

val_imgs = []
val_captions = []
for imgv in img_name_test_keys:
    capv_len = len(img_to_cap_vector_test[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector_test[imgv])


# In[79]:


len(train_imgs), len(train_captions), len(val_imgs), len(val_captions)


# In[80]:


def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = img / 255.
    caption = tokenizer(caption)
    return img, caption


# In[81]:


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_captions))

train_dataset = train_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[82]:


image_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ]
)


# In[172]:


import keras
from keras.saving import register_keras_serializable


# In[188]:


def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )
    inception_v3.trainable = False

    output = inception_v3.output
    output = tf.keras.layers.Reshape(
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model


# In[189]:


# @register_keras_serializable
class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")
    

    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)

        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training
        )

        x = self.layer_norm_2(x + attn_output)
        return x
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'embed_dim': self.embed_dim,
#             'num_heads': self.num_heads
#         })
#         return config


# In[190]:


# @register_keras_serializable
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'vocab_size': self.vocab_size,
#             'embed_dim': self.embed_dim,
#             'max_len': self.max_len
#         })
#         return config


# In[191]:


Embeddings(tokenizer.vocabulary_size(), EMBEDDING_DIM, MAX_LENGTH)(next(iter(train_dataset))[1]).shape


# In[192]:


# @register_keras_serializable
class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds


    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'embed_dim': self.embedding.token_embeddings.output_dim,
#             'units': self.ffn_layer_1.units,
#             'num_heads': self.attention_1.num_heads
#             # Add more configuration parameters if needed
#         })
#         return config


# In[193]:


# @register_keras_serializable
class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")


    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    

    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
    
    
#     def get_config(self):
#         """Returns the configuration of the model for serialization."""
#         config = super().get_config()  # Get base model config

#         # Get config of custom sub-modules (if applicable)
#         cnn_model_config = self.cnn_model.get_config() if self.cnn_model else None
#         encoder_config = self.encoder.get_config()
#         decoder_config = self.decoder.get_config()

#         # Update config with relevant attributes
#         config.update({
#             'cnn_model': cnn_model_config,
#             'encoder': encoder_config,
#             'decoder': decoder_config,
#             'image_aug': self.image_aug,
#         })
#         return config


# In[194]:


encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation
)


# In[195]:


cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)


# In[196]:


from tensorflow.keras.callbacks import ModelCheckpoint

caption_model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
#     callbacks=[early_stopping]
)

# Define the filepath and naming convention for the saved model
filepath = "/kaggle/working/image_captioning_best_model.keras"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)


# In[127]:


# caption_model.save(filepath)


# In[92]:


# idx2word(2).numpy().decode('utf-8')


# In[93]:


# def load_image_from_path(img_path):
#     img = tf.io.read_file(img_path)
#     img = tf.io.decode_jpeg(img, channels=3)
#     img = tf.keras.layers.Resizing(299, 299)(img)
#     img = tf.cast(img, tf.float32)
#     img = img / 255.
#     return img


# def generate_caption(img_path):
#     img = load_image_from_path(img_path)
#     img = tf.expand_dims(img, axis=0)
#     img_embed = caption_model.cnn_model(img)
#     img_encoded = caption_model.encoder(img_embed, training=False)

#     y_inp = '[start]'
#     for i in range(MAX_LENGTH-1):
#         tokenized = tokenizer([y_inp])[:, :-1]
#         mask = tf.cast(tokenized != 0, tf.int32)
#         pred = caption_model.decoder(
#             tokenized, img_encoded, training=False, mask=mask)
        
#         pred_idx = np.argmax(pred[0, i, :])
#         pred_word = idx2word(pred_idx).numpy().decode('utf-8')
#         if pred_word == '[end]':
#             break
        
#         y_inp += ' ' + pred_word
    
#     y_inp = y_inp.replace('[start] ', '')
#     return y_inp


# In[ ]:


# idx = random.randrange(0, len(val_imgs))
# img_path = val_imgs[idx]

# # pred_caption = generate_caption(img_path)
# print('Predicted Caption:', pred_caption)
# print()
# Image.open(img_path)


# In[2]:


# storing_trained_label = {}
# # for filepath in (train_df_image_text["label"])[400:404]:
#     storing_trained_label[f"/{filepath}"] = generate_caption(f"/{filepath}")
# storing_trained_label


# In[1]:


# url = "/kaggle/input/h-and-m-personalized-fashion-recommendations/images/057/0578433006.jpg"
# im = Image.open((url))
# im.save('tmp.jpg')

# # pred_caption = generate_caption('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/057/0578433006.jpg')
# print('Predicted Caption:', pred_caption)
# print()
# im


# # Saving and Loading Images

# In[137]:


# import os
# import shutil
# import pandas as pd

# def copy_files_to_new_folder(dataframe, source_column, destination_folder):
#     """
#     Copy files from the paths specified in a DataFrame column to a new folder.
    
#     Args:
#     - dataframe (DataFrame): DataFrame containing the file paths.
#     - source_column (str): Name of the column containing the file paths.
#     - destination_folder (str): Path to the destination folder where files will be copied.
    
#     Returns:
#     - None
#     """
#     # Create the destination folder if it doesn't exist
#     os.makedirs(destination_folder, exist_ok=True)
    
#     # Iterate through the file paths in the DataFrame column
#     for filepath in dataframe[source_column]:
#         # Extract the filename from the file path
#         filename = os.path.basename(f"/{filepath}")
        
#         # Construct the destination path
#         destination_path = os.path.join(destination_folder, filename)
        
#         # Copy the file to the destination folder
#         shutil.copyfile(f"/{filepath}", destination_path)

# # Example usage:
# # Assuming 'df' is your DataFrame with a column 'file_paths' containing the file paths,
# # and you want to copy the files to a folder named 'new_folder' in the current directory.
# copy_files_to_new_folder(train_df_image_text, 'label', '/kaggle/working/new_folder')


# In[138]:





# In[139]:





# In[ ]:




