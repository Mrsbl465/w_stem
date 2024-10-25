import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar el analizador de sentimientos de VADER
analyzer = SentimentIntensityAnalyzer()

# Lista de stopwords en inglés para eliminación adicional (puede ser personalizado)
stop_words = set(stopwords.words('english'))

def limpiar_urls(texto):
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    return texto

def limpiar_menciones(texto):
    texto = re.sub(r'@\w+', '', texto)
    return texto

def limpiar_numeros(texto):
    texto = re.sub(r'\d+', '', texto)
    return texto

def limpiar_puntuacion(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def limpiar_info_personal(texto):
    texto = re.sub(r'(DNI|número de teléfono|dirección)', '', texto, flags=re.IGNORECASE)
    return texto

def limpiar_stopwords(texto):
    palabras = word_tokenize(texto)
    palabras_limpias = [palabra for palabra in palabras if palabra.lower() not in stop_words]
    return ' '.join(palabras_limpias)

def limpiar_texto(texto):
    texto = limpiar_urls(texto)
    texto = limpiar_menciones(texto)
    texto = limpiar_numeros(texto)
    texto = limpiar_puntuacion(texto)
    texto = limpiar_info_personal(texto)
    texto = limpiar_stopwords(texto)
    return texto

def eliminar_retweets(tweet):
    if tweet.startswith('RT'):
        return None
    return tweet

def filtrar_tweet_por_longitud(tweet):
    palabras = word_tokenize(tweet)
    if len(palabras) < 3:
        return None
    return tweet

def filtrar_tweet_por_hashtags(tweet):
    if tweet.count('#') > 20:
        return None
    return tweet

def filtrar_tweets(tweet):
    tweet = eliminar_retweets(tweet)
    if tweet is None:
        return None
    tweet = filtrar_tweet_por_longitud(tweet)
    if tweet is None:
        return None
    tweet = filtrar_tweet_por_hashtags(tweet)
    return tweet

def analizar_sentimiento(texto):
    scores = analyzer.polarity_scores(texto)
    if scores['compound'] >= 0.05:
        return 'positivo'
    elif scores['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutral'

def preprocesar_y_etiquetar_tweets(tweets):
    tweets_procesados = []
    for tweet in tweets:
        tweet_limpio = limpiar_texto(tweet)
        tweet_filtrado = filtrar_tweets(tweet_limpio)
        if tweet_filtrado:
            sentimiento = analizar_sentimiento(tweet_filtrado)
            tweets_procesados.append((tweet_filtrado, sentimiento))
    return tweets_procesados

# Ejemplo de lista de tweets
tweets = [
    "¡Me encanta este lugar! #feliz",
    "Este lugar es terrible... #frustrado",
    "Visita https://example.com para más información.",
    "RT @usuario: No puedo creer lo que ha pasado...",
    "DNI: 12345678 Aquí está mi información personal."
]

# Preprocesar y etiquetar los tweets
tweets_procesados = preprocesar_y_etiquetar_tweets(tweets)

# Imprimir resultados detallados
def imprimir_resultados(tweets_procesados):
    for tweet, sentimiento in tweets_procesados:
        print(f"Tweet original procesado: {tweet}")
        print(f"Etiqueta de sentimiento: {sentimiento}\n")

imprimir_resultados(tweets_procesados)
