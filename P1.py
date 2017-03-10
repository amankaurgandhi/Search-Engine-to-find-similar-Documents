#############################
## Name: Aman Kaur Gandhi  ##
## Student ID: 1001164326  ##
#############################

#!/usr/bin/env python
import time
import os
import nltk
import math
import pprint
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
start_time = time.time()
stemmer = PorterStemmer()
wordcount = {}
postingList= {}
document_length= {}
idf = {}
corpusroot = './presidential_debates'      #path of the corpus/files
stopwords_list = set(stopwords.words('english'))	#extracting the stopwords
doc_files = os.listdir(corpusroot)	#files/documents in corpus
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')


def calculate_tfidf():
    """This function calculates the term frequency inverse document frequency"""
    N = 30
    for wd in wordcount:
        files = wordcount[wd].keys()
        for file in files:
            idfvalue = math.log10(N/len(files))
            wordcount[wd][file] = (1+ math.log10(wordcount[wd][file]))* idfvalue
            idf[wd] = idfvalue
			
def normalize_tfidfvector():
    """This function normalizes the tfidf vectors"""
    for file in doc_files:
        sum=0.0
        for w in wordcount:
            if file in wordcount[w]:
                sum += (wordcount[w][file] * wordcount[w][file])
        document_length[file]= math.sqrt(sum)
    for wrd in wordcount:
        postingList[wrd]=[]				#generates a posting list for each token in the corpus
        for document in wordcount[wrd]:
            wordcount[wrd][document]= wordcount[wrd][document]/ document_length[document]
            postingList[wrd].append((document, wordcount[wrd][document])) # appending the tuple of the form (document , TF-IDF weight) to the list
        postingList[wrd].sort(key=lambda x: x[1], reverse=True)	#sorting the posting list of each word in reverse order of their tfidf weights

def query(qstring):
    """This function returns a tuple in the form of (filename of the document, score)"""
    qstring = qstring.lower()			#converting the query to lower case
    queryCount={}
    cosine_similarity={}
    qlength = 0
    token = tokenizer.tokenize(qstring)    # tokenize the document to convert into tokens
    top10 = {}
    words = [stemmer.stem(word) for word in token if word not in stopwords_list]    #stemming the words that are not present in stopwords
    common_document= set()
    if len(words)>0:					#true means that there are more than 0 tokens in the query
        #finding the document which has all the tokens by using set intersection(common_document)
        for wr in words:
            if wr in postingList:
                top10[wr] = postingList[wr][:10]	#stores the top 10 from posting list of each token 
                l = set([x[0] for x in top10[wr]])	#gets the filename for the posting list of the token
                if len(common_document) == 0 :		
                    common_document = set(l)
                else:
                    common_document = set.intersection(common_document, l)  #checks for the document which has all the tokens of the query
            if wr in queryCount:
                queryCount[wr]+= 1
            else:
                queryCount[wr]= 1
        for key,value in queryCount.items(): #generating weighted query matrix
            queryCount[key] = 1 + math.log10(float(value))
        for file,count in queryCount.items():
            qlength += count * count
        qlength = math.sqrt(qlength)
        queryCount ={token: weight / qlength for token, weight in queryCount.items()} #normalizing the query vector
        for word in words:
            for d in doc_files:
                if word in top10:
                    tfidfscore= top10[word][-1][1]
                    for item in top10[word]:
                        if d==item[0]:
                            tfidfscore= item[1]
                else:
                    tfidfscore  = 0.00
                if d in cosine_similarity:
                    cosine_similarity[d] += queryCount[word] * tfidfscore
                else:
                    cosine_similarity[d] = queryCount[word] * tfidfscore
            #Finding the highest value of the cosinesimilarity          
            maximum = max(cosine_similarity, key=cosine_similarity.get) 
            if cosine_similarity[maximum] == 0.00:			#if true means that the tokens of the query are not present in the corpus
                    maximum = "None"
                    cosine_similarity[maximum] = 0
            else:                                          
                if maximum not in common_document:        #true means that there is no document which contains all the tokens of the query
                    maximum = "fetch more"
                    cosine_similarity[maximum] = 0
    else:									#if there are 0 tokens in the query			
        maximum = "None" 
        cosine_similarity[maximum] = 0
    return ( maximum, cosine_similarity[maximum])

def getweight(filename, token):
    """This function returns the tf idf weight of the token in given filename"""
    if token in wordcount:
        if filename in wordcount[token]:
            return wordcount[token][filename]
        else:
            return 0		#returns 0 if the token is not present in the filename provided
    else:
        return 0			#returns 0 if the token is not present in corpus

def getidf(token):     
    """This function returns the idf weight of the given token"""
    if token not in idf:
        return -1			#returns -1 if the token is not present in the corpus
    else:
        return idf[token]   #returns idf of the document if the token is present in the corpus  
  
for filename in doc_files:
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    #tokenized each document
    tokens = tokenizer.tokenize(doc)
    #stemming the words that are not present in stopwords
    words = [stemmer.stem(word) for word in tokens if word not in stopwords_list]
    #Calculating the term frequency for each word for each document in the corpus
    for word in words:
        if word in wordcount:
            if filename in wordcount[word]:
                wordcount[word][filename] += 1
            else:
                wordcount[word][filename] = 1
        else:
            wordcount[word]={}
            wordcount[word][filename]= 1
calculate_tfidf()
normalize_tfidfvector()
  
def main():  
    print("(%s, %.12f)" % query("health insurance wall street"))      
    print("(%s, %.12f)" % query("particular constitutional amendment"))
    print("(%s, %.12f)" % query("terror attack"))
    print("(%s, %.12f)" % query("vector entropy"))
    print("%.12f" % getweight("2012-10-03.txt","health"))
    print("%.12f" % getweight("1960-10-21.txt","reason"))
    print("%.12f" % getweight("1976-10-22.txt","agenda"))
    print("%.12f" % getweight("2012-10-16.txt","hispan"))
    print("%.12f" % getweight("2012-10-16.txt","hispanic"))
    print("%.12f" % getidf("health"))
    print("%.12f" % getidf("agenda"))
    print("%.12f" % getidf("vector"))
    print("%.12f" % getidf("reason"))
    print("%.12f" % getidf("hispan"))
    print("%.12f" % getidf("hispanic"))
    print(time.time()- start_time)
    
if __name__ == "__main__":
    main()                