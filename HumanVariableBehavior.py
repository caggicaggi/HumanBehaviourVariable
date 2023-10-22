import spacy
import text2emotion as t2e
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

#Load the advanced model of spacy
nlp = spacy.load('en_core_web_lg')

#Input sentence
sentence = "Did you see the message that went around about you, Marco? Francis sent it to everyone. I advise you not to show up tomorrow. You are really pathetic to think you could have kept it hidden."
#Sentence's analysis with spacy
doc = nlp(sentence)

#Sentiment analysis with text2emotion
def text2emotion_sentiment(text):
    emotions = t2e.get_emotion(text)
    dominant_emotion = max(emotions, key=emotions.get)
    return emotions, dominant_emotion

with open('C:\\Users\\caggi\\Desktop\\libreria.txt', "w", encoding="utf-8") as file:
    
    file.write(f"Original sentence: {sentence}\n\n")

    #Text2Emotion analysis
    te_emotions, te_dominant = text2emotion_sentiment(sentence)
    
    #Print detailed recognized emotions
    file.write("Sentiment Analysis with text2emotion:\n")
    for emotion, value in te_emotions.items():
        file.write(f"{emotion.capitalize()}: {value}\n")
    file.write(f"Dominant emotion with text2emotion: {te_dominant}\n")

    #Polarity and Subjectivity Analysis with TextBlob
    blob = TextBlob(sentence)
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
    vader_scores = analyzer.polarity_scores(sentence)
    file.write("\nSentiment Analysis with VADER from NLTK:\n")

    #Printing each score
    file.write(f"Negative Score: {round(vader_scores['neg'], 3)}\n")
    file.write(f"Neutral Score: {round(vader_scores['neu'], 3)}\n")
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
