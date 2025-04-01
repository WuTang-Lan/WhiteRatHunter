import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fake_news_dataset.csv")  # Contains 'text' and 'label' (1=Fake, 0=Real)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('model', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Test accuracy
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to classify a news article
def classify_news(article):
    prediction = pipeline.predict([article])
    return "Fake News" if prediction == 1 else "Real News"

# Example usage
article_text = "Breaking: Scientists discover a new planet that can support human life!"
print(classify_news(article_text))
