from  nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
from lesk import simple_lesk

syns = wordnet.synsets('Ravalish')
print(syns)
