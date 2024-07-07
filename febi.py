#import libray
import numpy as np
import pandas as pd
import regex as re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.models import load_model
import streamlit as st
import pickle

# model = load_model('./emoji_D2W4.keras')
path = './emoji_D2W4.pkl'

# Load the model from the pickle file
with open(path, 'rb') as f:
    loaded_model = pickle.load(f)
st.write("Hai")