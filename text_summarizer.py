from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import math

data = '''The most interesting fact about nature is the diversity of nature. It is the true beauty of nature. The diversity in nature has important role to play in the lifecycle on earth. The variety of life species on the earth is known as biodiversity of nature. It refers to the diversity of plants, animals, organisms, birds, bees etc. It also includes small microorganisms, fungi, algae, bacteria and many other insects and tiny creatures. It encompasses the diverse nature of ecosystem such as forests, desserts, mountains, rainforests and oceans. All are these are a part of nature.

Nature comprises of the connection between the life species, their habitat, their activities, living conditions and survival process. Biodiversity is measured higher in the tropical and floristic regions. The best studied species are mostly the large mammals. The diversity in nature cannot be measured in figures but all the living species large or tiny have an important role to play in nature. Biodiversity is responsible for balanced ecosystem. The spread of biodiversity varies across the globe depending on the soil, temperature, rainfall, altitude and geography.

Benefits of Biodiversity

More number of plant species provides us with greater variety of crops.
The diversity of species ensures the sustainability of all the species.
Healthy ecosystem can withstand several natural disasters.
Plants ensure remineralization and increase soil nutrients.
Plants provide us with food, medicinal resources, wood products, plants, diversity in genes and various species.
The social benefits of biodiversity are research, education, tourism, recreation and more.
It helps protect and preserve our water resources.
The biodiversity of organisms and fungi is important in the decomposing of waste.
Birds and butterflies play important role in seed dispersal and fertilization.
Biodiversity plays important role in absorbing and reducing pollution.
Conclusion

Biodiversity is valuable to humans. Biodiversity forms the base of infinite economic services that contribute to overall well-being of humans. Growth in population and economic development has led to inefficient use of biodiversity. Biodiversity has threats from human activities. Exploiting biodiversity can cause loss of species important for natural balance. Preservation and sustainable use of biodiversity is vital. Biodiversity is a gift from nature and we all should shield and protect it from further harm.'''

def summarize(data):
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(data)
    new_sent = ""
    for word in tokens:
        if(word not in stop_words):
            new_sent+=" "+word.lower()
    sent_tokens = sent_tokenize(new_sent) 

    new_sent_tokens = sent_tokenize(data)
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(sent_tokens)
    freq = cv_fit.toarray().sum(axis=0)/max(cv_fit.toarray().sum(axis=0))
    word = cv.get_feature_names_out()
    word2freq = dict(zip(word,freq))
    sent_list = []
    for i in sent_tokens:
        fre=0
        wordss = word_tokenize(i)
        for j in wordss:
            if(j in word):
                fre+=word2freq[j]
        sent_list.append(fre)   
    no_of_lines = math.ceil(len(sent_list)/2)
    brief_list = []
    for i in range(no_of_lines):
        brief_list.append(new_sent_tokens[sent_list.index(max(sent_list))])
        sent_list.pop(sent_list.index(max(sent_list)))


    return " ".join(brief_list[::-1])

print(data)
print('\n\n\n\n\n\n')

print(summarize(data))
