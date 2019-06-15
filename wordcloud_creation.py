import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

#Load the training set as a pandas dataframe
df = pd.read_csv("./train_set.csv",sep='\t')

#View the columns of the dataframe
df.columns

#Print some general info about the shape of the dataframe
print("There are {} articles and {} categories in this dataset. \n".format(df.shape[0],df.shape[1]))

#View for each article only its Content and the Category in which it belongs
df.loc[:, ['Content', 'Category']]

#Isolate the records of the dataframe that belong to the Business Category
business = df[df['Category'].isin(['Business'])]

#Isolate the Content of the records that belong to the Business Category
content_business = business.loc[:,'Content']

#Create a string containing all the Content of the records that belong to the Business Category
string_business = content_business.to_string()

#Generate a Wordcloud for the Business Category
wordcloud = WordCloud().generate(string_business)

#Plot the Business Wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Save the Business Wordcloud as a PNG 
wordcloud.to_file("business_wordcloud.png")

#Repeat the process for every other Category
technology = df[df['Category'].isin(['Technology'])]
content_technology = technology.loc[:,'Content']
string_technology = content_technology.to_string()
wordcloud = WordCloud().generate(string_technology)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("technology_wordcloud.png")

football = df[df['Category'].isin(['Football'])]
content_football = football.loc[:,'Content']
string_football = content_football.to_string()
wordcloud = WordCloud().generate(string_football)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("football_wordcloud.png")

politics = df[df['Category'].isin(['Politics'])]
content_politics = politics.loc[:,'Content']
string_politics = content_politics.to_string()
wordcloud = WordCloud().generate(string_politics)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("politics_wordcloud.png")

film = df[df['Category'].isin(['Film'])]
content_film = film.loc[:,'Content']
string_film = content_film.to_string()
wordcloud = WordCloud().generate(string_film)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("film_wordcloud.png")