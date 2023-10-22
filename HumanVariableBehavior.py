import spacy
import text2emotion as t2e
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json

#Load the advanced model of spacy
nlp = spacy.load('en_core_web_lg')

#DeepL Traduction
#Replace with your API key
DEEPL_API_KEY = ''
def translate_with_deepl(text, target_lang="EN"):
    base_url = "https://api-free.deepl.com/v2/translate"
    
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
        "User-Agent": "YourApp/1.0",
        "Content-Type": "application/json"
    }
    
    data = {
        "text": [text],
        "target_lang": target_lang
    }
    
    response = requests.post(base_url, headers=headers, data=json.dumps(data))
    
    # Print response status and content for debugging
    print("HTTP Status Code:", response.status_code)
    print("Response Content:", response.text)

    try:
        response_data = response.json()
    except ValueError:
        print("Decoding JSON has failed")
        return None, None  # Return None for both the translated text and detected language

    if 'translations' in response_data:
        translated_text = response_data['translations'][0]['text']
        detected_language = response_data['translations'][0]['detected_source_language']
        return translated_text, detected_language
    else:
        raise Exception("Error with translation.")

#Input sentence
original_sentence = "Hai visto il messaggio che girava su di te, Marco? Francis l'ha inviato a tutti. Ti consiglio di non presentarti domani. Sei davvero patetico a pensare di poterlo tenere nascosto."
# Nel tuo codice principale
translated_sentence, detected_language = translate_with_deepl(original_sentence)

#Analysis with spacy
doc = nlp(translated_sentence)

#Sentiment analysis with text2emotion
def text2emotion_sentiment(text):
    emotions = t2e.get_emotion(text)
    dominant_emotion = max(emotions, key=emotions.get)
    return emotions, dominant_emotion

with open('C:\\Users\\caggi\\Desktop\\libreria.txt', "w", encoding="utf-8") as file:
    
    file.write(f"Original sentence: {original_sentence}\n")
    file.write(f"Detected Language: {detected_language}\n")
    file.write(f"Translation: {translated_sentence}\n\n")
    
    #Text2Emotion analysis
    te_emotions, te_dominant = text2emotion_sentiment(translated_sentence)
    
    #Print detailed recognized emotions
    file.write("Sentiment Analysis with text2emotion:\n")
    for emotion, value in te_emotions.items():
        file.write(f"{emotion.capitalize()}: {value}\n")
    file.write(f"Dominant emotion with text2emotion: {te_dominant}\n")

    #Polarity and Subjectivity Analysis with TextBlob
    blob = TextBlob(translated_sentence)
    file.write("\nSentiment Analysis with TextBlob:\n")
    #Polarity
    polarity = round(blob.sentiment.polarity, 3)
    file.write(f"Polarity: {polarity}\n")
    if polarity > 0:
        file.write("Comment: The sentence has a positive tone.\n")
    elif polarity < 0:
        file.write("Comment: The sentence has a negative tone.\n")
    else:
        file.write("Comment: The sentence has a neutral tone.\n")

    #Subjectivity
    subjectivity = round(blob.sentiment.subjectivity, 3)
    file.write(f"Subjectivity: {subjectivity}\n")
    if subjectivity > 0.5:
        file.write("Comment: The sentence is subjective (based on personal opinions or feelings).\n")
    else:
        file.write("Comment: The sentence is objective (more fact-based).\n")

    #Sentiment Analysis with VADER from NLTK
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(translated_sentence)
    file.write("\nSentiment Analysis with VADER from NLTK:\n")

    #Printing each score
    file.write(f"Negative Score: {round(vader_scores['neg'], 3)}\n")
    file.write(f"Neutral Score:  {round(vader_scores['neu'], 3)}\n")
    file.write(f"Positive Score: {round(vader_scores['pos'], 3)}\n")
    file.write(f"Compound Score: {round(vader_scores['compound'], 3)}\n")

    #Interpreting the Compound Score
    if vader_scores['compound'] > 0.05:
        file.write("Comment: The sentence has a positive tone.\n")
    elif vader_scores['compound'] < -0.05:
        file.write("Comment: The sentence has a negative tone.\n")
    else:
        file.write("Comment: The sentence has a neutral tone.\n")

    #Print detailed recognized entities
    file.write("\nDetailed Recognized Entities with spaCy:\n")
    for ent in doc.ents:
        file.write(f"{ent.text} - {ent.label_} - Start: {ent.start_char}, End: {ent.end_char}\n")

    #Print lemmatization results with spaCy
    file.write("\nLemmatization with spaCy:\n")
    for token in doc:
        file.write(f"{token.text} ---> {token.lemma_}\n")

    #Print the dependency parsing tree and POS tagging
    file.write("\nDependency Parsing Tree + Part-of-Speech Tags with spaCy:\n")
    for token in doc:
        file.write(f"{token.text}({token.pos_}) <---{token.dep_}-- {token.head.text}({token.head.pos_})\n")
