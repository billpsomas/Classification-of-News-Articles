import nltk, csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from gensim.models.keyedvectors import KeyedVectors as wk
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Load Train
train_df = pd.read_csv('./train_set.csv', sep='\t')

stop_words = stopwords.words('english')

#vectorizer = CountVectorizer(stop_words=stop_words, max_features=10000)
#train_set_BoW = vectorizer.fit_transform(train_df.Content)
## Create an SVD transformed dataset containing 90% of variance.
#svd = TruncatedSVD(n_components=2200)
#train_df.Content = svd.fit_transform(train_df.Content)
#print('Total variance of : ', svd.explained_variance_ratio_.sum())

# Create Pipelines
svm_classifier_BoW = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), max_features=10000)),\
						   	   ('clf', SVC(kernel='linear', max_iter=200))])

random_forest_classifier_BoW = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), max_features=10000)),\
										 ('clf', RandomForestClassifier(n_estimators=100))])

svm_classifier_SVD = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), max_features=10000)),\
							   ('svd', TruncatedSVD(n_components=2200)),\
							   ('clf', SVC(kernel='linear', max_iter=200))])

random_forest_classifier_SVD = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'), max_features=10000)),\
									     ('svd', TruncatedSVD(n_components=2200)),\
									     ('clf', RandomForestClassifier(n_estimators=100))])

results = np.empty(shape=(3,7))
# Evaluation BoW
scores = cross_val_score(svm_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='accuracy')
print('SVM - BoW Accuracy : ', np.mean(scores))
results[0,0] = np.mean(scores)

scores = cross_val_score(svm_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='precision_micro')
print('SVM - BoW Precision : ', np.mean(scores))
results[1,0] = np.mean(scores)

scores = cross_val_score(svm_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='recall_micro')
print('SVM - BoW Recall : ', np.mean(scores))
results[2,0] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='accuracy')
print('Random Forest - BoW Accuracy : ', np.mean(scores))
results[0,1] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='precision_micro')
print('Random Forest - BoW Precision : ', np.mean(scores))
results[1,1] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_BoW, train_df.Content, train_df.Category, cv=10, scoring='recall_micro')
print('Random Forest - BoW Recall : ', np.mean(scores))
results[2,1] = np.mean(scores)

# Evaluation SVD
scores = cross_val_score(svm_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='accuracy')
print('SVM - SVD Accuracy : ', np.mean(scores))
results[0,2] = np.mean(scores)

scores = cross_val_score(svm_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='precision_micro')
print('SVM - SVD Precision : ', np.mean(scores))
results[1,2] = np.mean(scores)

scores = cross_val_score(svm_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='recall_micro')
print('SVM - SVD Recall : ', np.mean(scores))
results[2,2] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='accuracy')
print('Random Forest - SVD Accuracy : ', np.mean(scores))
results[0,3] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='precision_micro')
print('Random Forest - SVD Precision : ', np.mean(scores))
results[1,3] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_SVD, train_df.Content, train_df.Category, cv=10, scoring='recall_micro')
print('Random Forest - SVD Recall : ', np.mean(scores))
results[2,3] = np.mean(scores)

print('Loading Google\'s pretrained word2vector model')
model = wk.load_word2vec_format('./GoogleNews-vectors-negative300.bin', limit=500000, binary=True)
print('Loading completed')

print("Word2vector processing...")
w2v_vectors = np.empty((len(train_df), 300), float)
i = 0
for article in train_df.Content:
    article_average_vector = np.zeros(shape=(1, 300))
    word_count = 0
    for word in nltk.tokenize.word_tokenize(article):
        try:
            if word not in stop_words:
                word_count += 1
                article_average_vector = article_average_vector + model[word]
        except:
            continue
    article_average_vector = article_average_vector / (word_count + 1*(word_count==0))
    w2v_vectors[i, :] = article_average_vector
    i = i + 1

model.init_sims(replace=True)
model = None
print("Word2vector processing finished...")

svm_classifier_w2v = SVC(kernel='linear', max_iter=200)
random_forest_classifier_w2v = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(svm_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='accuracy')
print('SVM - w2v Accuracy : ', np.mean(scores))
results[0,4] = np.mean(scores)

scores = cross_val_score(svm_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='precision_micro')
print('SVM - w2v Precision : ', np.mean(scores))
results[1,4] = np.mean(scores)

scores = cross_val_score(svm_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='recall_micro')
print('SVM - w2v Recall : ', np.mean(scores))
results[2,4] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='accuracy')
print('Random Forest - w2v Accuracy : ', np.mean(scores))
results[0,5] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='precision_micro')
print('Random Forest - w2v Precision : ', np.mean(scores))
results[1,5] = np.mean(scores)

scores = cross_val_score(random_forest_classifier_w2v, w2v_vectors, train_df.Category, cv=10, scoring='recall_micro')
print('Random Forest - w2v Recall : ', np.mean(scores))
results[2,5] = np.mean(scores)

# Our Method
#class StemmedCountVectorizer(TfidfVectorizer):
#    def build_analyzer(self):
#        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

test_df = pd.read_csv("./test_set.csv",sep='\t')
train_set = train_df.Title + train_df.Content
test_set = test_df.Title + test_df.Content

articles = []
for article in train_set:
    article = article.split()
    words = []
    for word in article:
        tmp = ''.join(e.lower() for e in word if (e.isalnum() and not e.isdigit()))
        words.append(tmp)
    article = ' '.join(word for word in words)
    articles.append(article)

train_set = articles

articles = []
for article in test_set:
    article = article.split()
    words = []
    for word in article:
        tmp = ''.join(e.lower() for e in word if (e.isalnum() and not e.isdigit()))
        words.append(tmp)
    article = ' '.join(word for word in words)
    articles.append(article)

test_set = articles

#stemmer = SnowballStemmer("english", ignore_stopwords=True)
#stemmed_vect = StemmedCountVectorizer(stop_words='english', max_features=5000)
#train_set = stemmed_vect.fit_transform(train_set)
#test_set = stemmed_vect.fit_transform(test_set)

vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
train_set = vectorizer.fit_transform(train_set)
test_set = vectorizer.fit_transform(test_set)

#QDA_classifier = QuadraticDiscriminantAnalysis()
#scores = cross_val_score(QDA_classifier, w2v_vectors, train_df.Category, cv=10, scoring='accuracy')
#mlp_classifier = MLPClassifier(alpha=1)
#scores = cross_val_score(mlp_classifier, train_set, train_df.Category, cv=10, scoring='accuracy')

sgd_classifier = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-3, max_iter=30, tol=1e-3)

scores = cross_val_score(sgd_classifier, train_set, train_df.Category, cv=10, scoring='accuracy')
print('SVM with Stochastic Gradient Descent, Preprocessing & Tfidf - Accuracy : ', np.mean(scores))
results[0,6] = np.mean(scores)

scores = cross_val_score(sgd_classifier, train_set, train_df.Category, cv=10, scoring='precision_micro')
print('SVM with Stochastic Gradient Descent, Preprocessing & Tfidf - Precision : ', np.mean(scores))
results[1,6] = np.mean(scores)

scores = cross_val_score(sgd_classifier, train_set, train_df.Category, cv=10, scoring='recall_micro')
print('SVM with Stochastic Gradient Descent, Preprocessing & Tfidf - Recall : ', np.mean(scores))
results[2,6] = np.mean(scores)

# Save results in a tab separated csv
Measure = ['Accuracy', 'Precision', 'Recall']
with open('EvaluationMetric_10fold.csv', 'wt',  newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Statistic Measure'] + ['SVM (BoW)'] + ['Random Forest (BoW)'] + ['SVM (SVD)'] + ['Random Forest (SVD)'] + ['SVM (W2V)'] + ['Random Forest (W2V)'] + ['My Method'])
    for i in range(len(Measure)):
	    spamwriter.writerow([Measure[i], results[i,0], results[i,1], results[i,2], results[i,3], results[i,4], results[i,5], results[i,6]])


# Save the data in human readable form
# Create a Pandas dataframe from the data.
results = np.column_stack((Measure,results))
df = pd.DataFrame(results, columns=['Statistic Measure', 'SVM (BoW)', 'Random Forest (BoW)', 'SVM (SVD)', 'Random Forest (SVD)', 'SVM (W2V)', 'Random Forest (W2V)', 'My Method'])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('EvaluationMetric_10fold_for_humans.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')
writer.sheets['Sheet1'].set_column(1, 8, 20)

# Close the Pandas Excel writer and output the Excel file.
writer.save()


sgd_classifier.fit(train_set, train_df.Category)
predicts = sgd_classifier.predict(test_set)

with open('testSet_categories.csv', 'wt',  newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Test_Document_ID'] + ['Predicted_Category'] )
    for i in range(len(predicts)):
        spamwriter.writerow([test_df.Id[i], predicts[i]])