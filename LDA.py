import os
import spacy
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el modelo de lenguaje en inglés de spaCy
nlp = spacy.load('en_core_web_sm')

# Función para cargar archivos subidos
def load_uploaded_files(uploaded_files):
    corpus = []
    for uploaded_file in uploaded_files:
        # Leer el contenido del archivo cargado
        text = uploaded_file.read().decode('utf-8')
        corpus.append(text)
    return corpus

# Preprocesamiento de texto usando spaCy
def preprocess_text(text):
    # Procesar el texto con spaCy
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)  # Convertir la lista de palabras en una cadena

# Función para calcular la coherencia del modelo LDA
def compute_cosine_similarity(lda_model, vectorizer, num_topics):
    # Obtener las palabras más importantes por tema
    topic_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-10 - 1:-1]  # Top 10 palabras
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        topic_keywords.append(top_words)

    # Calcular la similitud del coseno entre los temas
    similarities = []
    for i in range(num_topics):
        for j in range(i+1, num_topics):
            # Comparar el conjunto de palabras de los dos temas
            words_i = " ".join(topic_keywords[i])
            words_j = " ".join(topic_keywords[j])
            vec_i = vectorizer.transform([words_i])
            vec_j = vectorizer.transform([words_j])
            sim = cosine_similarity(vec_i, vec_j)[0][0]
            similarities.append(sim)
    
    # Promediar la similitud para obtener una medida de la coherencia
    return np.mean(similarities)

# Función para elegir el número óptimo de temas basado en perplejidad y coherencia
def evaluate_optimal_topics(doc_term_matrix, min_topics=2, max_topics=10, coherence_weight=0.7, perplexity_weight=0.3, coherence_threshold=0.01):
    perplejidades = []
    coherencias = []
    
    for num_topics in range(min_topics, max_topics + 1):
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        
        # Evaluar perplejidad
        perplejidad = lda_model.perplexity(doc_term_matrix)
        perplejidades.append(perplejidad)
        
        # Evaluar coherencia utilizando similitud de coseno
        coherence = compute_cosine_similarity(lda_model, vectorizer, num_topics)
        coherencias.append(coherence)
        
        st.write(f"Evaluación para {num_topics} temas: Perplejidad = {perplejidad:.2f}, Coherencia = {coherence:.4f}")
    
    # Establecer un umbral de diferencia de coherencia para considerar que se ha estabilizado
    coherence_diffs = np.diff(coherencias)  # Calcular la diferencia entre coherencias consecutivas
    coherence_stable = [coherence_diffs[i] < coherence_threshold for i in range(len(coherence_diffs))]

    if any(coherence_stable):
        stable_index = coherence_stable.index(True)
        optimal_k = stable_index + min_topics + 1  # Obtener el índice donde se estabiliza la coherencia
        st.write(f"\n✅ Número óptimo de temas seleccionado basado en la estabilidad de coherencia: {optimal_k}")
    else:
        # Si no se estabiliza, seleccionar el k basado en la mejor coherencia
        optimal_k = np.argmax(coherencias) + min_topics
        st.write(f"\n✅ Número óptimo de temas basado en la evaluación combinada: {optimal_k}")
    
    # Graficar los resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.plot(range(min_topics, max_topics + 1), perplejidades, marker='o', color='b')
    ax1.set_title('Perplejidad por Número de Temas')
    ax1.set_xlabel('Número de Temas')
    ax1.set_ylabel('Perplejidad')

    ax2.plot(range(min_topics, max_topics + 1), coherencias, marker='o', color='r')
    ax2.set_title('Coherencia por Número de Temas')
    ax2.set_xlabel('Número de Temas')
    ax2.set_ylabel('Coherencia')
    
    st.pyplot(fig)
    
    return optimal_k

# Interfaz de Streamlit para cargar archivos
uploaded_files = st.file_uploader("Cargar archivos .txt", accept_multiple_files=True)

if uploaded_files:
    # Cargar y preprocesar el corpus
    raw_corpus = load_uploaded_files(uploaded_files)
    processed_corpus = [preprocess_text(doc) for doc in raw_corpus]

    # Vectorización usando CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    doc_term_matrix = vectorizer.fit_transform(processed_corpus)

    # Evaluar el número óptimo de temas
    optimal_num_topics = evaluate_optimal_topics(doc_term_matrix, min_topics=2, max_topics=18)

    # Entrenar el modelo LDA con el número óptimo de temas
    lda_model = LatentDirichletAllocation(n_components=optimal_num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    # Función para generar nubes de palabras para cada tema
    def generate_wordcloud_for_topics(lda_model, vectorizer, num_topics):
        for topic_idx, topic in enumerate(lda_model.components_):
            topic_words = {vectorizer.get_feature_names_out()[i]: topic[i] for i in range(len(vectorizer.get_feature_names_out()))}
            wordcloud = WordCloud(background_color='white').generate_from_frequencies(topic_words)
            plt.figure(figsize=(8, 6))
            plt.title(f"Word Cloud for Topic {topic_idx + 1}")
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()

    # Generar nubes de palabras para cada tema
    generate_wordcloud_for_topics(lda_model, vectorizer, optimal_num_topics)

