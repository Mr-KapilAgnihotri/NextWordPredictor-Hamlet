import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # ensures sequence length matches
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    
    # FIX 1: Add [0] to extract the integer from the numpy array
    predicted_word_index = np.argmax(predicted, axis=1)[0] 
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
            
    # FIX 2: Out-dent this so it only returns None if the loop finishes without a match
    return None     

st.title("Next word prediction with LSTM")
input_text = st.text_input("Enter the sequence of words")

if st.button("Predict Next Word"):
    # Assuming model.input_shape is (None, max_sequence_len - 1)
    max_sequence_len = model.input_shape[1] + 1 
    
    if input_text.strip(): # Good practice: check if input isn't just empty space
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word prediction: **{next_word}**')
    else:
        st.warning("Please enter some text first.")