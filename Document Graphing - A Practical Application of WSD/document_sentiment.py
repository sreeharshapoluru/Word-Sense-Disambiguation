import sentiment_mod as s
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
import time
import random


sample_text = state_union.raw("2005-GWBush.txt")
target_document = open("document.txt","r").read()
custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)
tokenized_document = custom_sent_tokenizer.tokenize(target_document)
random.shuffle(tokenized_document)


for l in tokenized_document:
    sentiment_value, confidence = s.sentiment(l)
    print(l, sentiment_value,confidence)
    if confidence*100 >=80:
            output = open("document-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
            time.sleep(0.3)

     
        
    
    
