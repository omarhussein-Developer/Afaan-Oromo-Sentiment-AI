mport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Odeeffannoo Fakkeenyaa (Sample Data)
# Ati gara fuulduraa faayilii CSV kee asitti galchita
data = {
    'yaada': [
        'Hojiin kun baay\'ee gaariidha', 'Baay\'ee natti tole', # Positive
        'Hojiin kun baay\'ee badaadha', 'Natti hin tolle',     # Negative
        'Guyyaa gaarii', 'Galatoomi',                         # Positive
        'Baay\'ee gadheedha', 'Hin xumurre'                    # Negative
    ],
    'gosa': ['gaarii', 'gaarii', 'badaa', 'badaa', 'gaarii', 'gaarii', 'badaa', 'badaa']
}

df = pd.DataFrame(data)

# 2. Barreeffama gara lakkoofsaatti jijjiiru (Vectorization)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['yaada'])

# 3. Model AI Leenjisuu
model = MultinomialNB()
model.fit(X, df['gosa'])

# 4. Model keenya madaaluu (Testing)
yaada_haaraa = ["Hojiin keessan baay'ee namatti tola"]
yaada_vector = vectorizer.transform(yaada_haaraa)
prediction = model.predict(yaada_vector)

print(f"Yaada: {yaada_haaraa[0]}")
print(f"AI'n akka jedhutti: {prediction[0]}")
