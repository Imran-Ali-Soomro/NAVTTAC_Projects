import streamlit as st
import string
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_option_menu import option_menu


st.title("💫Amazon IMBD Yelp Reviews💫")

st.write("## This is a web app to classify the sentiment of reviews from Amazon, IMBD and Yelp")

st.markdown("---")

selected = option_menu(
    menu_title='My App (Sagar Kumar) 💫',
    options= ['ABOUT APP', 'DATASET', 'PREDICITION', 'VISUALIZATION'],
    icons = ['book-half', 'file-earmark-code-fill', 'graph-up-arrow', 'pie-chart-fill'],
    
    menu_icon = 'toggle-on',
    default_index = 0,
    orientation = 'horizontal',
)

if selected == 'ABOUT APP':
    st.title(f' 💥 About App 💥')
    st.markdown('---')
    ## img1 = Image.open('ali.png')
    img2 = Image.open('pic1.png')
    col1, col2 = st.columns(2)
    with col2:
       st.image([img2], width=200)
    with col1:
        st.header('The app is created by Sagar Kumar')
        st.write('''This app will let you help to predict the sentiment of the review.
                 As there are many organizations that are taking reviews from the customers
                 because they want to know the feedback from their customers and users.
                 So, this app will help them to predict the sentiment of the review.
                 As there are many reviews that are positive and some are negative.
                 This App will help them to watch the overall reviews with the tag of
                 positive (1) and negetive (0) reviews with interacitve graphs.''')
        st.markdown('---')



