"""
https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs?resource=download
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""

###############################################
# PARSING
###############################################

import pandas as pd

#/Users/paulux/Documents/M1/TATIA/PROJET/fake_job_postings.csv
mydata = pd.read_csv("fake_job_postings.csv")  # On stocke le fichier csv dans un dataframe
df = mydata.sort_values('fraudulent', ascending=False) # On trie par ordre decroissant le dataframe par rapport a la colonne fraudulent  Cette etape servira pour la selection d'un jeu de donnees plus petit sans perdre de cas frauduleux
df.drop(columns=df.columns[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True) # On enleve les colonnes autres que le job_id, la description et la classe frauduleux

df = df.dropna()

#desc = df['description']
desc = [str(x) for x in df['description']]
#fraud = df['fraudulent']
fraud = [int(x) for x in df['fraudulent']]
sumf=0 #Pour connaitre le nombre de cas frauduleux
for i in fraud:
    sumf+=i
print("sumf=", sumf)

#df.drop(df.index[4*sumf:], inplace=True) # Taille echantillon  : 3460

#Les deux lignes ci dessous : Taille echantillon : 50
df.drop(df.index[10:sumf], inplace=True)
df.drop(df.index[50:], inplace=True)

print(df.shape)
#print(df.isna().sum())

# Injection des descriptions dans des fichiers

path = 'PROJET/desc/'
for i in range(len(desc)):
    filename = 'desc' + str(i) + '.txt'
    if fraud[i]:  # A ranger dans le dossier fraud
        file = open(path + 'fraud/' + filename, "a", encoding='utf-8')
    else:  # A rander dans le dossier acc
        file = open(path + 'acc/' + filename, "a", encoding='utf-8')
    file.write(desc[i])
    file.close()

###############################################
# VECTORISATION
###############################################

# TRAINING

# INIT
import sklearn
from sklearn.datasets import load_files
train = load_files('C:\\Users\\mylye\\PROJET\\desc', allowed_extensions=['.txt'])

# TOKENIZING
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.data)
X_train_counts.shape

# OCC TO FREQ
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

count_vect.vocabulary_.get(u'skill')

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# TRAINING A CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train.target)

docs_new = ['die', 'Do YOU have the sales skills or entrepreneurial drive to join us?FlexKom GmbH is a German company experiencing massive success in Europe and the UK with a revolutionary Customer Loyalty program. And FlexKom is opening for business in the USA as FlexKom America Inc.!Instead of consumers carrying multiple Rewards cards, they carry an app on their phone. ONE APP (or flexkom rewards card) Universally accepted with small to medium businesses. The customer gets cash back and points on every purchase at a FlexKom participating merchant, and the merchant turns his advertising cost into an INCOME STREAM.Not just a good idea. It\'s GENIUS combined with technology that didn\'t exist five years ago.Are YOU an TOP-Level Sales Professional? Are you a Serial Entrepreneur? We are presently aggressively seeking sales pros and entrepreneurs all over the USA to join our team and bring this disruptive technology to the US Market. Do YOU have what it takes to build a sales organization of your own? We are seeking Entrepreneurs who are open to new business opportunities. FlexKom Associates are Global Team Members (GTM), independent business owners operating under IRS W-9 regulations.Watch this short intro video then contact me for more information!']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
     print('%r => %s' % (doc, train.target_names[category]))