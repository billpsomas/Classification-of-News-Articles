import csv, sys
import gensim.utils
import pandas as pd
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity

if len(sys.argv)!=2:
	print('You must enter a similarity threshold as an argument during the execution command i.e. python duplicates.py 0.7')
	exit()
similarity = float(sys.argv[1])

df = pd.read_csv('./train_set.csv', sep='\t')

dictionary = Dictionary([simple_preprocess(article) for article in df.Content])
corpus = [dictionary.doc2bow(simple_preprocess(article)) for article in df.Content]

tfidf = TfidfModel(corpus)

index_temp = get_tmpfile("index")
index = Similarity(index_temp, tfidf[corpus], num_features=len(dictionary))

duplicate_count = 0
duplicates = []

for i, s in enumerate(index):
    for j, similarity_value in enumerate(s):
        if similarity_value >= similarity and i < j:
            duplicate_count += 1
            duplicates.append([i, j, similarity_value])

print("Found:", duplicate_count, "duplicates.")

# Save results in a tab separated csv
with open('duplicatePairs.csv', 'wt',  newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Document_ID1'] + ['Document_ID2'] + ['Similarity'])
    for i in range(duplicate_count):
	    spamwriter.writerow([duplicates[i][0], duplicates[i][1], duplicates[i][2]])


# Save the data in human readable form
# Create a Pandas dataframe from the data.
df = pd.DataFrame(duplicates, columns=['Document ID1', 'Document ID2', 'Similarity'])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('duplicate_pairs_for_humans.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')
writer.sheets['Sheet1'].set_column(1, 4, 20)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

