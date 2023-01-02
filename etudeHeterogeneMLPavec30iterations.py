"""
https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs?resource=download
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""
import pandas as pd
import os
import glob
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

###############################################
# PARSING
###############################################

def count_fraud(fraud_content):
    sumf = 0  # Pour connaitre le nombre de cas frauduleux
    for i in fraud_content:
        sumf += i
    return sumf

tabSize = []
tabAccurancyMLP = []

plus = 1 # Sert a augmenter la taille de l'echantillon

############################################################
# AUGMENTATION DE LA TAILLE DE L'ECHANTILLON : 
# SEUL LES ACCEPTABLES SONT RAJOUTES
# LES FRAUDULEUX N'AUGMENTENT PAS
############################################################


while (plus <=5):
    mydata = pd.read_csv("fake_job_postings.csv")  # On stocke le fichier csv dans un dataframe
    df = mydata.sort_values('fraudulent', ascending=False) # On trie par ordre decroissant le dataframe par rapport a la colonne fraudulent  Cette etape servira pour la selection d'un jeu de donnees plus petit sans perdre de cas frauduleux
    df.drop(columns=df.columns[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True) # On enleve les colonnes autres que le job_id, la description et la classe frauduleux

    df = df.dropna()

    fraud = [int(x) for x in df['fraudulent']]
    sumf = count_fraud(fraud)

    #Taille echantillon varie
    df.drop(df.index[sumf + int(sumf*plus):], inplace=True) # On fait augmenter ici la taille de l'echantillon
    desc = [str(x) for x in df['description']]
    fraud = [int(x) for x in df['fraudulent']]
    tabSize.append(len(fraud))
    # Injection des descriptions dans des fichiers
    path = 'desc/'

    # Clean folders
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
    train = load_files('desc', allowed_extensions=['.txt'])

    # TOKENIZING
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train.data)
    X_train_counts.shape

    # OCC TO FREQ
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

    # MLPCLASSIFIER
    clf = MLPClassifier(max_iter=30).fit(X_train_tfidf, train.target)  # On change ici le nombre d'iterations maximale
    docs_new = ["Great Home Health Opportunity for  Physical Therapists, and Occupational Therapists! Good Life Home Care, an established, family-owned and Medicare certified home health agency seeks clinicians with experience in the home health setting to support with treatment visits and case management responsibilities in the Watsonville and Santa Cruz area.  Good Life is known for its commitment to excellence in patient care and supportive work environment. Service territory is flexible to meet the needs of our employees. POSITION DESCRIPTIONProvide patient care on a per visit basisWork with administrative and supervisory personnel regarding therapy visits to assure high quality and proper follow-up patient careParticipate in case conferences to ensure optimum communication within and between departments and to discuss active issuesReport all events that vary from policies and procedures and/or standards of therapy care to the Therapy SupervisorFollow physician orders for treatment",
                'Do YOU have the sales skills or entrepreneurial drive to join us?FlexKom GmbH is a German company experiencing massive success in Europe and the UK with a revolutionary Customer Loyalty program. And FlexKom is opening for business in the USA as FlexKom America Inc.!Instead of consumers carrying multiple Rewards cards, they carry an app on their phone. ONE APP (or flexkom rewards card) Universally accepted with small to medium businesses. The customer gets cash back and points on every purchase at a FlexKom participating merchant, and the merchant turns his advertising cost into an INCOME STREAM.Not just a good idea. It\'s GENIUS combined with technology that didn\'t exist five years ago.Are YOU an TOP-Level Sales Professional? Are you a Serial Entrepreneur? We are presently aggressively seeking sales pros and entrepreneurs all over the USA to join our team and bring this disruptive technology to the US Market. Do YOU have what it takes to build a sales organization of your own? We are seeking Entrepreneurs who are open to new business opportunities. FlexKom Associates are Global Team Members (GTM), independent business owners operating under IRS W-9 regulations.Watch this short intro video then contact me for more information!#URL_670b1628db01f0732bdd9bc819fbb121f093e4cb0248dd5bae5953cffa4b2efd#Tyler Hollinger#EMAIL_f575c1f8220f5b937bc4ab50f82740211a306d0f5fbb07fe03c31b21a9891a52# #PHONE_dfc9b369cea27d543c03683680e08ee4144de0e06bd70922e0eae6c375bc9328#']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    pred_train_lrsgd = clf.predict(X_train_tfidf)
    tabAccurancyMLP.append(metrics.accuracy_score(train.target, pred_train_lrsgd))


    plus+=1


#######################################################
# TRACAGE DES COURBES POUR TOUS LES ALGORITHMES 
#######################################################

plt.plot(tabSize, tabAccurancyMLP)
plt.xlabel('Taille echantillon')
plt.ylabel('Accuracy_score')
plt.title('MLP avec max_iter=30')
plt.show()
