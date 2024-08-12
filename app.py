import tensorflow as tf
import gradio as gr
import re
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


tema = gr.themes.Soft()

MAX_SEQUENCE_LENGTH = 50

# Carica il modello di analisi del sentiment
model = tf.keras.models.load_model('./Prova_best_model.hdf5')

with open('./tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Funzione per preprocessare il testo
def preprocess_text(text, stem=False):
    stemmer = SnowballStemmer('english')
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9']+"

    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    preprocess = " ".join(tokens)

    sequences = tokenizer.texts_to_sequences([preprocess])
    p_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return p_text

# Funzione per predire il sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]
    elif isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()[0]
    print("Valore di prediction associato a ", text, ' :', prediction)
    sentiment = 'POSITIVE :)' if prediction > 0.5 else 'NEGATIVE :('
    return sentiment


# Funzione per formattare l'output
def format_output(text):
    sentiment = predict_sentiment(text)
    color = 'green' if sentiment == 'POSITIVE :)' else 'red'
    return f"<span style='color:{color}; font-size: 80px;'>{sentiment}</span>"


# Crea l'interfaccia Gradio
iface = gr.Interface(
    fn=format_output,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence...", label="Input Text"),
    outputs=gr.HTML(label="Sentiment"),
    title="Sentiment Analysis",
    description="Enter a sentence to analyze its sentiment.",
    theme=tema
)

if __name__ == "__main__":
    iface.launch(server_port=7872)
