#importing essential libraries and modules.
from  nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
from lesk import simple_lesk

# input query from user
input_query = input("Enter the query:",)
print("You entered the following query: \n", input_query)

# tokenizing the input query
tokenized_query = word_tokenize(input_query)
print(" The tokenized query is as follows: \n", tokenized_query)

#improving the stopwords
improved_stopwords = set(stopwords.words("english"))
#adding custom words
improved_stopwords.update(("0","1","2","3","4","5","6","7","8","9",".","..","...","/","'","@","!","#","$","%","^","&","*","(",")","[","]","{","}",":",";",'''"''',",",
                           "<",">","?","`","~","-","_","=","+","|","a","about","above","accordance","according","accordingly","across","actually","after","afterwards",
                           "again","all","almost","along","already","also","although","always","am","among","an","another","any","anybody","anyhow","anymore","anyone",
                           "anything","anyway","anywhere","are","aren't","around","as","aside","at","away","b","became","because","become","been","before","beforehand",
                           "behind","below","beside","beyond","between","both","bottom","but","by","c","come","can't","could","couldn't","d","did","didn't","do","does",
                           "doesn't","e","either","else","elsewhere","every","everybody","everything","everywhere","f","few","for","from","front","further","get","h","had",
                           "has","hasn't","have","haven't","here","herself","him","himself","how","however","i","i.e","if","i'll","i'm","in","indeed","into","instead","is",
                           "isn't","it","it'll","it's","itself","i've","j","just","k","l","let","m","maybe","most","much","must","my","mr","mrs","n","nobody","none","nor",
                           "noting","o","of","off","often","ok","okay","onto","only","or","our","ought","ourselves","p","q","r","s","say","seem","self","shall","shall",
                           "she'll","should","shouldn't","some","somebody","somehow","someone","something","sometime","somewhat","somewhere","soon","sorry","such","t","thank",
                           "that","that'll","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","thereof",
                           "there're","there's","thereto","thereupon","there've","these","they","they're","they've","this","those","thou","though","through","throughtout","thru",
                           "till","to","toward","u","under","unless","unlike","until","unto","up","upon","v","via","viz","w","was","wasn't","we","were","weren't","what","whatever",
                           "when","whence","where","whereas","whereby","wherein","wherever","whereupon","whether","which","while","who","whoever","whole","whose","why","with",
                           "within","without","wont't","would","wouldn't","x","y","you","you'll","your","your's","yourself","yourselves","you've","z"))


# filtering the stopwords
stop_words = set(stopwords.words("english"))
filtered_query = []

for t in tokenized_query:
    if wordnet.synsets(t) != []:
        if (nltk.pos_tag([t])[0][1]) != "NNP":
            if t.lower() not in improved_stopwords:
                filtered_query.append(t)

print("The filtered query after removing the stop words is as follows :\n", filtered_query)

# printing the sysnset and the meaning
for f in filtered_query:
    print('The differnet forms and their meanings of "', f.upper(), '" are as follows : \n')
    syns = wordnet.synsets(f)
    i=1
    for s in syns:
        print(i,".""   ""Form : ", s, "\n   Meaning : ",s.definition())
        i=i+1
    print('*'*20)

# using the LESK algorithm to disambiguate the meaning of the word using the context.
print('Using the simple lesk algorithm to disambiguate the word sense')
print(' '*100)
for f in filtered_query:
    print('The correct sense of the word "',f.upper(),' "is as follows:')
    print('-'*50)
    print('Context:',input_query)
    correct_sense = simple_lesk(input_query,f)
    print('Sense:',correct_sense)
    print('Definition:',correct_sense.definition())
    print(' '*100)

    


                 


