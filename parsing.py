"""
https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs?resource=download
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""

import pandas as pd
import os
import glob

###############################################
# PARSING
###############################################


def count_fraud(fraud_content):
    sumf = 0  # accumulator to know the number of fraudulent contents
    for fc in fraud_content:
        sumf += fc
    return sumf


my_data = pd.read_csv("fake_job_postings.csv")  # Stock csv file in dataframe

# Order by DESC dataframe compared to fraud column
# This step is mandatory to retrieve smaller play set
df = my_data.sort_values('fraudulent', ascending=False)

# Remove any column different of job_id, description and fraudulent class
df.drop(columns=df.columns[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True)
df = df.dropna()


#desc = df['description']
desc = [str(x) for x in df['description']]
#fraud = df['fraudulent']
fraud = [int(x) for x in df['fraudulent']]

sumf = count_fraud(fraud)

print("Il y a", sumf, "cas frauduleux")
#df.drop(df.index[4*sumf:], inplace=True) # Sample length  : 3460

# Two lines below: Sample length : 50
df.drop(df.index[10:sumf], inplace=True)
df.drop(df.index[50:], inplace=True)

# Inject descriptions in files
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
    if fraud[i]:  # Arrange in fraud folder
        file = open(path + 'fraud/' + filename, "a", encoding='utf-8')
    else:  # Arrange in acc folder
        file = open(path + 'acc/' + filename, "a", encoding='utf-8')
    file.write(desc[i])
    file.close()

print("Parsing finish")
