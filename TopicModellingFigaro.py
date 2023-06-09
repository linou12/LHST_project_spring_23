#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling 

# In[2]:


import tomotopy as tp
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randrange
import spacy
import unidecode
import helpers


# In[2]:


DATA_PATH = "/scratch/students/bousbina/corpus/France/data/future_articles_fr.csv"
KW = "futur" # without accent
JOURNAL = "Figaro"
START_DATE = pd.to_datetime("1840-01-01") ## Articles on this date WILL be included
END_DATE = pd.to_datetime("1920-01-01") ## Articles on this date WON't be included 
SPACY_LANGUAGE = "fr_core_news_sm" 
COUNTRY = "fr"


# In[3]:


data = pd.read_csv(DATA_PATH )


# In[4]:


def preprocess_text(sp, text: str, user_data = None): ### takes text as a string (not list) and return list
    text = text.lower()
    text = [word.lemma_ for word in sp(text) if word.is_alpha and (not word.is_stop) and len(word)>3]  
    ### above: removes punctuation, digits, stop words, lemmatizes words
    return text


# In[5]:


def create_corpus(data, language):
    raw_articles = data["content"].to_list()
    sp = spacy.load(language, disable=["ner",  "entity_linker",   "parser", 
                                           "textcat", "textcat_multilabel",  "senter",  "sentencizer",  "transformer"
                                          ]) ## you can load any language now and it will automatically decide on the stop words
    sp.max_length = 2327128
    start= time.time()
    raw_docs = []
    num_articles = len(raw_articles)
    for i, doc in enumerate(raw_articles):
        if i%500 == 1:
            print("Runtime: %.2f seconds" %(time.time() - start), "|| Completed: %s of %s" %(i, num_articles))
        raw_docs.append(preprocess_text(sp, doc))

    corpus = tp.utils.Corpus()
    for doc in raw_docs:
        if doc:
            corpus.add_doc(doc)
            
    return raw_docs, corpus


# In[6]:


print("Create corpus for tp model...")
_, corpus = create_corpus(data, SPACY_LANGUAGE) #.sample(2000)
print("Done !")


# In[3]:


############################################
###### You should change only this
k1_max = 3 ###(min 1) if k1_max - 1 == best k1
k2_max = 25
### k2 should always be higher than k1 (min 1)
####################
params = []
scores = []
num_iter = 0
max_iter = 0
for k1 in range(1, k1_max):
    for k2 in range(k1,k2_max):
        max_iter +=1
        
start = time.time()
for k1 in range(1, k1_max):
    for k2 in range(k1,k2_max):
        num_iter+=1
        model = tp.PAModel(tw=tp.TermWeight.IDF, min_df=10, rm_top=10, k1=k1, k2=k2, corpus=corpus, seed=0)
        model.train(500, workers=1)
        score = tp.coherence.Coherence(model, coherence="c_v").get_score() #try with 'u_mass' instead of 'c_v' 
        params.append((k1,k2))
        scores.append(score)
        print("Runtime: %.2f seconds" %(time.time() - start), "|| Number of Searches: %s out of  %s" %(num_iter, max_iter))


# In[ ]:


best_params = np.argmax(scores)
print("(BEST MODEL) k1=%s k2=%s coherence=%.2f"  %(params[best_params][0], params[best_params][1],  scores[best_params]))

data = pd.DataFrame(data={'k1':list(zip(*params))[0], 'k2':list(zip(*params))[1], 'score':scores})
data = data.pivot(index='k1', columns='k2', values='score')
sns.heatmap(data)
plt.show()


# ### 1. Choose the best parameters

# In[ ]:


#################################################
### Set the best model
best_k1 = params[best_params][0] 
best_k2 = params[best_params][1]
model = tp.PAModel(tw=tp.TermWeight.IDF, min_cf=5, k1=best_k1, k2=best_k2, corpus=corpus, seed=0)
model.burn_in=100
model.train(1000, workers=1)


# ### 2. Print All the topics

# In[ ]:


with open("TopicModelling_Figaro_futur.txt", "w") as file:
    for k in range(best_k2):
        file.write('Topic #{}\n'.format(k))
        file.write("\t" + " ".join([w for w, _ in model.get_topic_words(k)]) + "\n")


# In[ ]:




