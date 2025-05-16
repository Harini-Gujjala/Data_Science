import pandas as pd
import numpy as np
import re
import string
from google.colab import files
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

"""Upload the File"""

print("Labelled_real_and_Fake_Dataset.csv")
uploaded = files.upload()
df = pd.read_csv("Labelled_real_and_Fake_Dataset.csv")

"""Handling Missing Data

"""

print("Missing values:\n", df.isnull().sum())

"""Filling or Removing Duplicates"""

# Drop duplicate rows based on 'title' and 'text'
df.drop_duplicates(subset=['title', 'text'], inplace=True)

""" Standardize Text Format"""

# Convert to lowercase, remove numbers and punctuation
def standardize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

"""Tokenization & Lemmatization using NLTK"""

# Define function for full text cleaning
def preprocess_text(text):
    text = standardize_text(text)  # Step 3
    words = text.strip().split()  # Tokenize
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize
    return ' '.join(words)

"""Preprocessing"""

# Apply preprocessing to the 'text' column and create a new 'cleaned_text' column
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Optional: Apply preprocessing to the 'title' column (if needed)
df['cleaned_title'] = df['title'].apply(preprocess_text)

# Preview the result (show the original 'title' and the 'cleaned_text' columns)
df[['title', 'cleaned_text']].head()

""" Exploratory Data Analysis (EDA)"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Distribution of Real vs Fake News')
plt.xticks([0, 1], ['Fake', 'Real'])
plt.xlabel('News Type')
plt.ylabel('Count')
plt.show()

# Calculate article lengths and store in a new column
df['article_length'] = df['text'].apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='article_length', hue='label', bins=50, kde=True)
plt.title('Article Length Distribution by Label')
plt.xlabel('Article Length')
plt.ylabel('Count')
plt.legend(['Fake', 'Real'])
plt.show()

from wordcloud import WordCloud

# Join texts by class
fake_text = ' '.join(df[df['label'] == 'REAL']['cleaned_text'])
real_text = ' '.join(df[df['label'] == 'FAKE']['cleaned_text'])

# Generate wordclouds
wordcloud_fake = WordCloud(width=400, height=300).generate(fake_text)
wordcloud_real = WordCloud(width=400, height=300).generate(real_text)

# Plot them
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title('Fake News WordCloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title('Real News WordCloud')
plt.axis('off')

plt.show()

df['label'].value_counts().plot.pie(labels=['REAL', 'FAKE'], autopct='%1.1f%%', colors=['green', 'red'])
plt.title("Real vs Fake News Proportion")
plt.ylabel("")
plt.show()

sns.boxplot(x='label', y='article_length', data=df)
plt.xticks([0, 1], ['FAKE', 'REAL'])
plt.title("Article Length by News Type")
plt.xlabel("News Type")
plt.ylabel("Article Length")
plt.show()

# Calculate article lengths and store in a new column called 'text_length'
df['text_length'] = df['text'].apply(len)

# Convert 'label' column to numerical representation (0 for FAKE, 1 for REAL)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

sns.heatmap(df[['article_length', 'text_length', 'label']].corr(), annot=True, cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

"""Text Vectorization (TF-IDF)"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine title and text if both exist
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text']
elif 'title' in df.columns:
    df['content'] = df['title']
else:
    raise ValueError("No 'title' or 'text' column found in the dataset.")

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf_vectorizer.fit_transform(df['content'])
print(X_tfidf.shape)

""" text length features"""

# Create text length features
df['title_length'] = df['title'].fillna('').apply(len)
df['text_length'] = df['text'].fillna('').apply(len)
df['content_length'] = df['content'].apply(len)
print(df[['title_length', 'text_length', 'content_length']].head())

!pip install vaderSentiment

"""sentiment analysis"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each article's text
df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Show a few examples
print(df[['label', 'sentiment']].head())

# Calculate average sentiment scores for fake and real articles
avg_sentiment = df.groupby('label')['sentiment'].mean()
print("\nAverage Sentiment Scores:\n", avg_sentiment)

# Visualize sentiment distribution
df.boxplot(column='sentiment', by='label', grid=False)
plt.title('Sentiment Distribution by Label')
plt.suptitle('')
plt.xlabel('News Label')
plt.ylabel('Compound Sentiment Score')
plt.show()

# Create new text length-based features
df['text_length'] = df['text'].apply(len)
df['title_length'] = df['title'].apply(len)
df['num_words_text'] = df['text'].apply(lambda x: len(x.split()))
df['num_words_title'] = df['title'].apply(lambda x: len(x.split()))

# Print the updated DataFrame with new features
print(df[['text_length', 'title_length', 'num_words_text', 'num_words_title']])

print(df['label'].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])  # FAKE=0, REAL=1 (typically)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# Assuming X_tfidf is your feature matrix and df['label'] is your target variable
X = X_tfidf
y = df['label']

# Example using oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# For undersampling (optional alternative)
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)

import numpy as np
unique, counts = np.unique(y_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 3. Train and Evaluate Models
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test) # Changed Y_test to y_test
print('Accuracy score of the test data : ', test_data_accuracy)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train) # Changed Y_train to y_train
print('Accuracy score of the training data : ', training_data_accuracy)

print(y_test.iloc[3])  # Accessing the 4th element of y_test Series using .iloc

X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')
