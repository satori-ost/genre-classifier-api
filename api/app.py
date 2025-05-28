# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3, joblib, pickle, numpy as np
from src.preprocess import Preprocessor  
from gensim.models import FastText

# Шляхи до артефактів
DB_PATH       = 'data/songs.db'
VECTORIZER_FP = 'models/tfidf_vectorizer.pkl'
FT_MODEL_FP   = 'models/fasttext.model'
SVM_MODEL_FP  = 'models/svm_genre.pkl'
RF_MODEL_FP   = 'models/rf_genre.pkl'

# Ініціалізація
app = FastAPI()
tfidf = joblib.load(VECTORIZER_FP)
ft    = FastText.load(FT_MODEL_FP)

class Request(BaseModel):
    lyrics: str

class Response(BaseModel):
    genre: str
    probability: float

@app.post("/predict", response_model=Response)
def predict(req: Request):
    # Препроцесінг
    pp = Preprocessor()
    lem = pp.lemmatize(pp.clean(req.lyrics))
    # Фічі
    v1 = tfidf.transform([lem]).toarray()[0]
    v2 = np.mean([ft.wv[w] for w in lem.split() if w in ft.wv], axis=0)
    vec = np.hstack([v1, v2])
    # Завантажуємо модель
    clf = joblib.load(SVM_MODEL_FP)
    genre = clf.predict([vec])[0]
    prob  = float(max(clf.decision_function([vec])[0]))
    return {"genre": genre, "probability": prob}
