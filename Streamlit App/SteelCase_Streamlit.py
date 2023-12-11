# Imports
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

#-------------------------------------------------------------------------------------------
#Data
@st.cache_data
def read_data():
    data = pd.read_csv('floorplan_dataset_valid.csv')
    image_paths = data['image_path'].tolist()
    return data, image_paths

@st.cache_data
def get_embeddings():
    embeddings_df = pd.read_csv('Embeddings_valid.csv', index_col=0)
    embed_df_nump = embeddings_df.to_numpy()
    return embed_df_nump

@st.cache_data#(allow_output_mutation=True)
def init_knn(embed_df_nump):
    knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    knn.fit(embed_df_nump)
    return knn

@st.cache_data#(allow_output_mutation=True)
def init_vgg19():
    base_model = VGG19(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return model

data, image_paths = read_data()
embed_df_nump = get_embeddings()
knn = init_knn(embed_df_nump)
model = init_vgg19()

#-------------------------------------------------------------------------------------------
# Functions
# Get embedding using vgg19 model
def get_embedding(model, pil_img):
    if pil_img.size != (224, 224):
        pil_img = pil_img.resize((224, 224))
    img_data = np.array(pil_img)
    img_data = preprocess_input(img_data)

    embedding_vector = model.predict(np.expand_dims(img_data, axis=0))

    embedding_vector_flatten = embedding_vector.flatten()
    return embedding_vector_flatten

# Get Input image embedding
def get_user_image_embedding(image_path, model):
    user_image = Image.open(image_path)
    user_image_mod = user_image.resize((224,224))
    user_embedding = get_embedding(model, user_image_mod)
    return user_embedding

def calculate_nearest_neighbors(user_embedding, knn_model, image_paths, num_neighbors=5):
    # Find nearest neighbors
    distances, indices = knn_model.kneighbors([user_embedding], n_neighbors=num_neighbors + 1)
    neighbor_indices = indices[0][1:]
    nearest_images = [image_paths[i] for i in neighbor_indices]
    return nearest_images
    
#-------------------------------------------------------------------------------------------------------------------------------------------
#Streamlit Application
st.markdown("<h1 style='text-align: center; color: white;'>Steelcase Floorplan Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>MSDS Capstone Project</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Patrick Govan, Harish Neelam, Yan Lyu</h2>", unsafe_allow_html=True)
    
col1, col2 = st.columns(2)

with col1:
    st.header('Image Input')
    input_image = st.file_uploader('Upload a .PNG')
    st.text('For Demo, use dropdown below')
    input_image_demo = st.selectbox('Please select an image',
                                   ('Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'),
                                   index=None, placeholder='Please select an image')
    if input_image_demo == 'Image 1':
        img = r'Total_Images\0a7cbdf5b54148238346a02797264298\0a7cbdf5b54148238346a02797264298_Top.png'
    if input_image_demo == 'Image 2':
        img = r'Total_Images\0a14edbeff34447a88acda602f872614\0a14edbeff34447a88acda602f872614_Top.png'
    if input_image_demo == 'Image 3':
        img = r'Total_Images\00deb8f9f33341609e147736dc7a6c7a\00deb8f9f33341609e147736dc7a6c7a_Top.png'
    if input_image_demo == 'Image 4':
        img = r'Total_Images\0a979f0550304ea6b6d83d785d0691a0\0a979f0550304ea6b6d83d785d0691a0_Top.png'
    if input_image_demo == 'Image 5':
        img = r'Total_Images\0a5958254a414d5a92f98d6e0ecef715\0a5958254a414d5a92f98d6e0ecef715_Top.png'
    #if input_image_demo == None:
        #break
        
    st.image(img, caption = 'Original Image')
        
    # Generate embedding for the user image
    user_embedding = get_user_image_embedding(img, model)
    
        
    
with col2:
    st.header('Nearest Neighbor Images Output')
    nearest_images = calculate_nearest_neighbors(user_embedding, knn, image_paths)
    idx = 1
    for image_path in nearest_images:
        st.image(image_path, use_column_width=True, caption="Nearest Neighbor {}".format(idx))
        idx += 1
