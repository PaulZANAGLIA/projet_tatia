"""
https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs?resource=download
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""

###############################################
# PARSING
###############################################

import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt

def count_fraud(fraud_content):
    sumf = 0  # Pour connaitre le nombre de cas frauduleux
    for i in fraud_content:
        sumf += i
    return sumf

mydata = pd.read_csv("fake_job_postings.csv")  # On stocke le fichier csv dans un dataframe
df = mydata.sort_values('fraudulent', ascending=False) # On trie par ordre decroissant le dataframe par rapport a la colonne fraudulent  Cette etape servira pour la selection d'un jeu de donnees plus petit sans perdre de cas frauduleux
df.drop(columns=df.columns[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True) # On enleve les colonnes autres que le job_id, la description et la classe frauduleux

df = df.dropna()

print('PETITE SELECTION HOMOGENE')
print('### Avant selection ###')

fraud = [int(x) for x in df['fraudulent']]
sumf = count_fraud(fraud)

print('df.shape =', df.shape)
print('Il y a', sumf, 'cas frauduleux.')

#Les deux lignes ci dessous : Taille echantillon : 20
df.drop(df.index[10:sumf], inplace=True)
df.drop(df.index[20:], inplace=True)

print('###\nApres selection ###')
print('df.shape =', df.shape)
print('Il y a',count_fraud(df['fraudulent']), 'cas frauduleux')

desc = [str(x) for x in df['description']]
fraud = [int(x) for x in df['fraudulent']]

print('len fraud =',len(fraud))
print('Il y a',count_fraud(fraud), 'cas frauduleux.')

# Injection des descriptions dans des fichiers
path = 'desc/'

# Clean folders
import os
import glob

acc_files = glob.glob(path + 'acc/*.txt')

for f in acc_files:
    os.remove(f)

fraud_files = glob.glob(path + 'fraud/*.txt')
for f in fraud_files:
    os.remove(f)

for i in range(len(desc)):
    filename = 'desc' + str(i) + '.txt'
    if fraud[i]:  # A ranger dans le dossier fraud
        file = open('desc/fraud/' + filename, "a", encoding='utf-8')
        #print('fraud')
    else:  # A rander dans le dossier acc
        file = open('desc/acc/' + filename, "a", encoding='utf-8')
    file.write(desc[i])
    file.close()

print("\nParsing finish\n")

###############################################
# VECTORISATION
###############################################

# INIT
import sklearn
from sklearn.datasets import load_files
train = load_files('desc', allowed_extensions=['.txt'])

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

###############################################
# TRAINING
###############################################

print('\nALGORITHME MLPCLASSIFIER')
# TRAINING A CLASSIFIER
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter=80).fit(X_train_tfidf, train.target)

docs_new = ["Great Home Health Opportunity for  Physical Therapists, and Occupational Therapists! Good Life Home Care, an established, family-owned and Medicare certified home health agency seeks clinicians with experience in the home health setting to support with treatment visits and case management responsibilities in the Watsonville and Santa Cruz area.  Good Life is known for its commitment to excellence in patient care and supportive work environment. Service territory is flexible to meet the needs of our employees. POSITION DESCRIPTIONProvide patient care on a per visit basisWork with administrative and supervisory personnel regarding therapy visits to assure high quality and proper follow-up patient careParticipate in case conferences to ensure optimum communication within and between departments and to discuss active issuesReport all events that vary from policies and procedures and/or standards of therapy care to the Therapy SupervisorFollow physician orders for treatment",
            'Do YOU have the sales skills or entrepreneurial drive to join us?FlexKom GmbH is a German company experiencing massive success in Europe and the UK with a revolutionary Customer Loyalty program. And FlexKom is opening for business in the USA as FlexKom America Inc.!Instead of consumers carrying multiple Rewards cards, they carry an app on their phone. ONE APP (or flexkom rewards card) Universally accepted with small to medium businesses. The customer gets cash back and points on every purchase at a FlexKom participating merchant, and the merchant turns his advertising cost into an INCOME STREAM.Not just a good idea. It\'s GENIUS combined with technology that didn\'t exist five years ago.Are YOU an TOP-Level Sales Professional? Are you a Serial Entrepreneur? We are presently aggressively seeking sales pros and entrepreneurs all over the USA to join our team and bring this disruptive technology to the US Market. Do YOU have what it takes to build a sales organization of your own? We are seeking Entrepreneurs who are open to new business opportunities. FlexKom Associates are Global Team Members (GTM), independent business owners operating under IRS W-9 regulations.Watch this short intro video then contact me for more information!#URL_670b1628db01f0732bdd9bc819fbb121f093e4cb0248dd5bae5953cffa4b2efd#Tyler Hollinger#EMAIL_f575c1f8220f5b937bc4ab50f82740211a306d0f5fbb07fe03c31b21a9891a52# #PHONE_dfc9b369cea27d543c03683680e08ee4144de0e06bd70922e0eae6c375bc9328#']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# Voir ACCURACY
pred_train_lrsgd = clf.predict(X_train_tfidf)
print('Logistic Regression with MLPCLASSIFIER Model training dataset accuracy: {0:0.4f}'. format(metrics.accuracy_score(train.target, pred_train_lrsgd)))
print('\nClassification report')
contentMPL = classification_report(train.target, pred_train_lrsgd)
print(contentMPL)
fichier = open("Classification_report_petite_MPL80.txt", "a")
fichier.write(contentMPL)
fichier.close()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
skplt.metrics.plot_confusion_matrix(train.target, pred_train_lrsgd,
                                    normalize=True,
                                    title="Petite selection MPL 80 iterations",
                                    cmap="Purples",
                                    ax=ax
                                    )
plt.show()