import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
import re

class PoemRecommender:

    def __init__(self, poems_dataset, vectorizer_path='pickles_files/vectorizer.pickle', matrix_path='pickles_files/poems_count_matrix.pickle'):
        self.poems_dataset = poems_dataset
        self.lemmatizer = WordNetLemmatizer()
        
        base_path = os.path.dirname(__file__)
        vectorizer_full_path = os.path.join(base_path, vectorizer_path)
        matrix_full_path = os.path.join(base_path, matrix_path)
        
        try:
            with open(vectorizer_full_path, 'rb') as file:
                self.count_vectorizer = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de vectorizador en {vectorizer_full_path}. Por favor, asegúrese de que el archivo exista.")

        try:
            with open(matrix_full_path, 'rb') as file:
                self.poems_count_matrix = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de matriz de conteo en {matrix_full_path}. Por favor, asegúrese de que el archivo exista.")

    def get_synonyms(self, word):
        blob_word = Word(word)
        synonyms = blob_word.synsets
        synonyms_list = []
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonyms_list.append(lemma.name())
        return set(synonyms_list)

    def get_synonyms_tokenized(self, text):
        tokens = TextBlob(text).words    
        total_words = set()
        for word in tokens:
            synonyms = self.get_synonyms(word)
            if synonyms:
                total_words.update(synonyms)
            else:
                total_words.add(word)
        return ' '.join(total_words)

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def recommend_poems(self, theme, top_n=3):
        # Preprocesa el tema y expande con sinonimos
        expanded_theme = self.get_synonyms_tokenized(theme)
        processed_theme = self.preprocess_text(expanded_theme)
        
        # Transforma el tema procesado
        theme_count = self.count_vectorizer.transform([processed_theme])
        
        # Calcula las similitudes coseno entre el tema y todos los poemas
        cosine_similarities = cosine_similarity(theme_count, self.poems_count_matrix).flatten()
        
        # Obtiene los indices de los top_n poemas más similares
        top_poem_indices = cosine_similarities.argsort()[-top_n:][::-1]
        
        # Selecciona los poemas recomendados
        recommended_poems = self.poems_dataset.iloc[top_poem_indices]
        
        return recommended_poems
    
    def recommend_continuation(self, current_text, top_n=1):
        # Preprocesa el texto actual
        processed_text = self.preprocess_text(current_text)
        
        # Transforma el texto procesado
        text_count = self.count_vectorizer.transform([processed_text])
        
        # Calcula las similitudes coseno entre el texto y todos los poemas
        cosine_similarities = cosine_similarity(text_count, self.poems_count_matrix).flatten()
        
        # Obtiene el índice de los top_n poemas más similares
        top_poem_indices = cosine_similarities.argsort()[-14:][::-1]
        
        # Lista de palabras vacías que queremos omitir
        stopwords = {'in', 'the', 'at', 'and', 'a', 'of', 'to', 'as', 'both', 'not', 'be', 'but', 'or', 'on', 'for', 'with', 'by', 'is', 'was', 'were', 'that', 'this', 'an', 'it'}

        # Función para encontrar la última palabra significativa (no stopword)
        def find_last_significant_word(text):
            text_blob = TextBlob(text)
            for word in text_blob.words[::-1]:
                if word.lower() not in stopwords:
                    return word
            return None

        last_noun = find_last_significant_word(current_text)
        
        if last_noun:
            # Extrae la posible continuación de cada poema
            recommended_phrases = []
            next_word = []
            for poem in self.poems_dataset.iloc[top_poem_indices]['poema']:
                sentences = TextBlob(poem).sentences
                for sentence in sentences:
                    if last_noun in sentence.words:
                        recommended_phrases.append(sentence)
                        # Si la palabra es la última de la frase se agrega el '.'
                        if last_noun == sentence.words[-1]:
                            next_word.append('.')
                        else:
                            next_word.append(sentence.words[sentence.words.index(last_noun) + 1])
                        
            # Si se encontraron frases recomendadas
            if recommended_phrases:
                # Obtener las frecuencias de las palabras siguientes
                freq = {}
                for word in next_word:
                    if word in freq:
                        freq[word] += 1
                    else:
                        freq[word] = 1                
                # Obtengo la palabra siguiente más frecuente
                next_word_suggestions = max(freq, key=freq.get)

                # Busco las frases que contengan la palabra siguiente más frecuente
                highlighted_phrases = []
                for phrase, word in zip(recommended_phrases, next_word):
                    if word == next_word_suggestions:
                        # Resalta el último sustantivo y la palabra siguiente en la frase
                        highlighted_phrase = re.sub(r'\b' + re.escape(last_noun) + r'\b', f"<span class='highlight-noun'>{last_noun}</span>", str(phrase))
                        highlighted_phrase = re.sub(r'\b' + re.escape(word) + r'\b', f"<span class='highlight-next'>{word}</span>", highlighted_phrase)
                        highlighted_phrases.append(highlighted_phrase)

                # Creo un diccionario con las frases y las palabras siguientes
                frase_and_next_word = {
                    'frases': highlighted_phrases,
                    'palabra_siguiente': next_word_suggestions
                }

                # Retorno el diccionario y el last noun
                return frase_and_next_word, last_noun

            else:
                # quitamos todas las ocurrencias de last_noun del texto actual
                current_text = ' '.join([word for word in current_text.split() if word.lower() != last_noun.lower()])
                # encontramos la última palabra significativa
                last_noun = find_last_significant_word(current_text)
                # si encontramos un last_noun
                if last_noun:
                    # llamamos recursivamente a la función
                    return self.recommend_continuation(current_text, top_n)
                else:
                    return {}, 'No se encontró frase'
        else:
            return {}, 'No se encontró sustantivo en el texto actual'




