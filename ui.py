import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_excel('./List_Emoji.xlsx')

def replace_emoji_with_meaning(text):
    def tambah_spasi_emoji(text):
        pattern = re.compile(r'(\p{So})')
        separated_text = re.sub(pattern, r' \1 ', text)
        return separated_text

    if isinstance(text, str):
        text_with_spaces = tambah_spasi_emoji(text)
        final_text = []
        seen_emojis = set()  # Untuk melacak emoji yang sudah ditemukan
        for word in text_with_spaces.split():
            word = str(word)
            found = data[data['emoji'] == word]
            if not found.empty:
                emoji_meaning = found['makna'].values[0]
                # Memeriksa jika emoji sudah ditemukan sebelumnya atau tidak
                if word not in seen_emojis:
                    final_text.append(emoji_meaning)
                    seen_emojis.add(word)  # Menambahkan emoji ke set yang sudah ditemukan
            else:
                final_text.append(word)
        return ' '.join(final_text)
    else:
        return text
    
def cleaning(text):
  # menghapus tab, new line, dan back slice
  text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
  # menghapus mention, link, hashtag
  text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
  # menghapus incomplete URL
  text =  text.replace("http://", " ").replace("https://", " ")
  #menghapus nomor
  text = re.sub(r"\d+", "", text)
  #menghapus punctuation
  text = text.translate(str.maketrans("","",string.punctuation))
  #menghapus spasi tunggal dan ganda
  text = re.sub('\s+',' ',text)
  # menghapus kata 1 abjad
  text=re.sub(r"\b[a-zA-Z]\b", "", text)
  return text

def word_tokenizing(text):
    return word_tokenize(text)

# membuat dict
kamus = pd.read_excel('./kbba.xlsx')
slanglist=kamus['slang']
bakulist=kamus['baku']
kbba={}
for i in range (len(slanglist)):
  kbba[slanglist[i]]=bakulist[i]

def normalisasi(tokens):
    hasil_normalisasi = []
    for kata in tokens:
        if kata in kbba.keys():
            hasil_normalisasi.append(kbba[kata])
        else:
            hasil_normalisasi.append(kata)
    return hasil_normalisasi

from nltk.corpus import stopwords

def remove_stopwords(x):
  hasil = []
  negasi = ['tidak', 'bukan', 'tanpa']

  list_stopwords = stopwords.words('indonesian')
  #Mengubah List ke dictionary
  stoplist = set(list_stopwords)
  # memisahkan antar kata berdasarkan spasi
  for i in x :
    # dilakukan pengecekan pada stoplist
    if i not in stoplist or i in negasi:
      # memasukkan kata yang tidak ada di stoplist ke array
      hasil.append(i)
  return hasil

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

Fact = StemmerFactory()
Stemmer = Fact.create_stemmer()

def stemming(x):
  hasil = []
  for kata in x:
    a = Stemmer.stem(kata)
    hasil.append(a)
  return hasil

df = pd.read_excel("./emoji_unicode_pre.xlsx")
df = df.dropna()
df = df.reset_index(drop=True)
stem = df['Stemming']
# # Membersihkan dan membagi setiap kalimat menjadi daftar token
sentences = [sentence.replace("[", "").replace("]", "").replace("'", "").replace(",", "").split() for sentence in stem]


# Tokenisasi teks
tok = Tokenizer()
tok.fit_on_texts(sentences)
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(sentences)

# # Padding sequence
max_rev_len = max(len(sentence) for sentence in sentences)
pad_rev = pad_sequences(encd_rev, maxlen=max_rev_len, padding='post')


# Memuat model dari file .h5
model = load_model('./emoji.keras')

st.title('Pengaruh Emoji Pada Analisis Sentimen Ulasan Pantai Jawa Timur Menggunakan :red[Skip-gram dan Long Short-Term Memory (LSTM)]')

input_text = st.text_area('Masukkan teks:')
if st.button('Prediksi',type="primary" ):
    teks_konversi = replace_emoji_with_meaning(input_text)
    # st.write(teks_konversi)
    teks_CharacterCleansing = cleaning(teks_konversi)
    # st.write(teks_CharacterCleansing)
    teks_caseFolding = teks_CharacterCleansing.lower()
    # st.write(teks_caseFolding)
    teks_tokenisasi = word_tokenizing(teks_caseFolding)
    # st.write(teks_tokenisasi)
    teks_normalisasi = normalisasi(teks_tokenisasi)
    # st.write(teks_normalisasi)
    teks_stopword = remove_stopwords(teks_normalisasi)
    # st.write(teks_stopword)
    teks_stemming = stemming(teks_stopword)
    # st.write(teks_stemming)
    # Lakukan tokenisasi pada teks yang telah diproses
    encoded_text = tok.texts_to_sequences([teks_stemming])

    # Lakukan padding pada teks yang telah ditokenisasi
    padded_text = pad_sequences(encoded_text, maxlen=max_rev_len, padding='post')
    # padded_text

    # Lakukan prediksi menggunakan model yang telah Anda muat sebelumnya
    predictions = model.predict(padded_text)

    # Ambil label prediksi (diasumsikan bentuk keluaran model adalah one-hot encoded)
    predicted_label = np.argmax(predictions, axis=1)

    if predicted_label == 0 :
       st.write(f'Ulasan memiliki <span style="background-color:red; padding: 5px; border-radius: 5px; color:white;">*Sentimen Negatif*</span>', unsafe_allow_html=True)
    else :
      st.write(f'Ulasan memiliki <span style="background-color:blue; padding: 5px; border-radius: 5px; color:white;">*Sentimen Positif*</span>', unsafe_allow_html=True)