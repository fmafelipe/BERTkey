import numpy as np
import pandas as pd
import streamlit as st
from keybert import KeyBERT
import re


st.set_page_config("KeyBert", page_icon="👹")

idioma = st.sidebar.radio(
    "Escoja el idioma",
    ["Ingles","Español"],
    help = "Segun el idioma se escoge el modelo para generar el embedding de documento"
    )
    
if idioma == "Ingles":
    @st.cache(allow_output_mutation=True)
    def load_model():
        return KeyBERT(model='all-MiniLM-L6-v2')
        
    kw_model = load_model()

else:
    @st.cache(allow_output_mutation=True)
    def load_model():
        return KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        
    kw_model = load_model()
    
numero_result = st.sidebar.slider("Numero de resultados",1,30,1)    

min_ngram = st.sidebar.number_input("Min_ngram", 1,4,1, help="Numero minimo de palabras en un resultado")
max_ngram = st.sidebar.number_input("Max_ngram",  1,4,1, help="Numero maximo de palabras en un resultado")

if min_ngram > max_ngram:
    st.warning("min_ngram>max_ngram se debe cumplir")
    st.stop()

stop_words = st.sidebar.checkbox('Stop Words', help="Solo en ingles")
if stop_words:
    words= 'english'
else:
    words = None
    
mmr = st.sidebar.checkbox('MMR',help="Para usar esta opcion solo se debe seleccionar esta checkbox")
if mmr:
    mmr1 = True
    words = None
else:
    mmr1 = False
    
Diversity = st.sidebar.number_input('Diversity',0.0,1.0,0.10,0.1, help="Valor de la diversidad para diversificar los resultados")


st.title("Aplicacion en streamlit aplicando KeyBERT")

st.write("Idioma escogido", idioma)

with st.form(key="my_form"):
    
    doc = st.text_area('Ingrese el texto, maximo 500 palabras', height=410)
    num_words = len(re.findall(r"\w+", doc))
    st.write('numero de palabras',num_words)
    submit_button = st.form_submit_button(label="✨ Obtener resultados")
    
    
max_words = 500
if num_words > max_words:
    st.write("El texto que introdujo es de mas de 500 palabras, se tomaran las primeras 500")
    doc = doc[:max_words]
    
if not submit_button:
    st.stop()
    
k_words = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_ngram,max_ngram),
    use_mmr= mmr1,
    stop_words = words,
    top_n = numero_result,
    diversity = Diversity,
    )

resaltado = kw_model.extract_keywords(
    doc,
    highlight=True
    )


st.header("Resultados")

df = pd.DataFrame(
    k_words,
    columns = ["Palabra/frase","relevancia"],
    )

df.sort_values(by='relevancia',ascending=False)
df

