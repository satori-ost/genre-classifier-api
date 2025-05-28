#!/usr/bin/env python3
# main.py — покращений варіант

import os
import sys
import sqlite3
import pandas as pd
import spacy
import pickle
import joblib
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from gensim.models import FastText

# ——— Конфігурація ————————————————————————————————————————
DB_PATH         = 'data/songs.db'
CSV_PATH        = 'data/genredata.csv'
VECTORIZER_FP   = 'models/tfidf_vectorizer.pkl'
FT_MODEL_FP     = 'models/fasttext.model'
SVM_MODEL_FP    = 'models/svm_genre.pkl'
RF_MODEL_FP     = 'models/rf_genre.pkl'

# ——— Ініціалізація БД ———————————————————————————————————
def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL COLLATE NOCASE,
        artist TEXT NOT NULL COLLATE NOCASE,
        year INTEGER,
        genre TEXT NOT NULL COLLATE NOCASE,
        lyrics_raw TEXT NOT NULL,
        lyrics_clean TEXT
      );
      CREATE TABLE IF NOT EXISTS features (
        song_id INTEGER PRIMARY KEY REFERENCES songs(id) ON DELETE CASCADE,
        tfidf BLOB NOT NULL,
        embedding BLOB NOT NULL
      );
      CREATE TABLE IF NOT EXISTS results (
        song_id INTEGER REFERENCES songs(id) ON DELETE CASCADE,
        predicted_genre TEXT,
        probability REAL,
        timestamp TEXT DEFAULT (datetime('now','localtime')),
        PRIMARY KEY(song_id, timestamp)
      );
    """)
    conn.commit()
    conn.close()
    print("✔ База даних готова:", DB_PATH)

# ——— Імпорт CSV → songs —————————————————————————————
def import_csv():
    # 1) Читаємо CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Підключаємося до БД
    conn = sqlite3.connect(DB_PATH)
    # дістаємо вже існуючі id
    existing = pd.read_sql("SELECT id FROM songs;", conn)['id'].tolist()
    # фільтруємо вхідний df
    df_new = df[~df['id'].isin(existing)]

    if df_new.empty:
        print("ℹ️  Нових записів для імпорту не знайдено.")
    else:
        # 3) Імпортуємо тільки нові рядки
        df_new[['id','title','artist','year','genre','lyrics_raw']]\
            .to_sql('songs', conn, if_exists='append', index=False)
        print(f"✔ Імпортовано {len(df_new)} нових рядків у songs")

    conn.close()
# ——— Препроцесінг тексту —————————————————————————————
class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('uk_core_news_lg', disable=['parser','ner','textcat'])
    def clean(self, text):
        return text.replace('\n',' ').strip()
    def lemmatize(self, text):
        doc = self.nlp(text)
        return ' '.join(tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space)
    def run_all(self):
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT id, lyrics_raw FROM songs WHERE lyrics_clean IS NULL;")
        rows = cur.fetchall()
        for sid, raw in rows:
            lem = self.lemmatize(self.clean(raw))
            cur.execute("UPDATE songs SET lyrics_clean = ? WHERE id = ?;", (lem, sid))
        conn.commit(); conn.close()
        print(f"✔ Препроцесінг застосовано до {len(rows)} пісень")

# ——— Екстракція ознак ———————————————————————————————
class FeatureBuilder:
    def fit(self):
        # 1) Завантажуємо тексти
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT id, lyrics_clean FROM songs;", conn)
        conn.close()
        corpus = df['lyrics_clean'].fillna("").tolist()

        # 2) TF–IDF з ngram_range=(1,3)
        tfidf = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1,3))
        X_tfidf = tfidf.fit_transform(corpus)

        # 3) FastText
        tokenized = [t.split() for t in corpus]
        ft = FastText(vector_size=100, window=5, min_count=2)
        ft.build_vocab(tokenized)
        ft.train(tokenized, total_examples=len(tokenized), epochs=5)

        # 4) Зберігаємо вектори-моделі
        os.makedirs('models', exist_ok=True)
        joblib.dump(tfidf, VECTORIZER_FP)
        ft.save(FT_MODEL_FP)

        # 5) Пишемо ознаки в БД
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        for i, sid in enumerate(df['id']):
            vec_tf = X_tfidf[i].toarray()[0]
            vec_ft = np.mean([ft.wv[w] for w in tokenized[i]], axis=0)
            cur.execute(
                "INSERT OR REPLACE INTO features VALUES (?,?,?)",
                (int(sid),
                 sqlite3.Binary(pickle.dumps(vec_tf)),
                 sqlite3.Binary(pickle.dumps(vec_ft)))
            )
        conn.commit(); conn.close()
        print("✔ Ознаки збережено в features")

# ——— Навчання і порівняння з GridSearch —————————————————
class GenreClassifier:
    def __init__(self):
        self.tfidf = joblib.load(VECTORIZER_FP)
        self.fasttext = FastText.load(FT_MODEL_FP)

    def load_features(self):
        conn = sqlite3.connect(DB_PATH)
        df_f = pd.read_sql("SELECT song_id, tfidf, embedding FROM features;", conn)
        df_s = pd.read_sql("SELECT id, genre FROM songs;", conn)
        conn.close()
        X1 = np.vstack(df_f['tfidf'].apply(lambda b: pickle.loads(b)).values)
        X2 = np.vstack(df_f['embedding'].apply(lambda b: pickle.loads(b)).values)
        X = np.hstack([X1, X2])
        y = df_s.set_index('id').loc[df_f['song_id'], 'genre'].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_compare(self):
        X_train, X_test, y_train, y_test = self.load_features()

        # --- SVM з class_weight і GridSearchCV ---
        base_svm = LinearSVC(class_weight='balanced', max_iter=10000)
        params_svm = {'C':[0.01,0.1,1,10]}
        gs_svm = GridSearchCV(base_svm, params_svm, cv=5, scoring='f1_macro')
        gs_svm.fit(X_train, y_train)
        best_svm = gs_svm.best_estimator_
        y_svm = best_svm.predict(X_test)
        print("=== Best SVM (class_weight, ngram 1–3) ===")
        print("Params:", gs_svm.best_params_)
        print(classification_report(y_test, y_svm))

        # --- RF з class_weight і GridSearchCV ---
        base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        params_rf = {'n_estimators':[50,100,200], 'max_depth':[None,10,20]}
        gs_rf = GridSearchCV(base_rf, params_rf, cv=5, scoring='f1_macro')
        gs_rf.fit(X_train, y_train)
        best_rf = gs_rf.best_estimator_
        y_rf = best_rf.predict(X_test)
        print("=== Best RF (class_weight, GridSearch) ===")
        print("Params:", gs_rf.best_params_)
        print(classification_report(y_test, y_rf))

        # Зберігаємо обидві
        joblib.dump(best_svm, SVM_MODEL_FP)
        joblib.dump(best_rf, RF_MODEL_FP)
        print(f"✔ Кращі моделі збережено:\n  - {SVM_MODEL_FP}\n  - {RF_MODEL_FP}")

    def predict(self, text, model='svm'):
        pp = Preprocessor()
        lem = pp.lemmatize(pp.clean(text))
        v1 = self.tfidf.transform([lem]).toarray()[0]
        v2 = np.mean([self.fasttext.wv[w] for w in lem.split()], axis=0)
        vec = np.hstack([v1, v2])

        if model=='svm':
            clf = joblib.load(SVM_MODEL_FP)
            score = max(clf.decision_function([vec])[0])
        else:
            clf = joblib.load(RF_MODEL_FP)
            score = float(max(clf.predict_proba([vec])[0]))

        return clf.predict([vec])[0], score

# ——— CLI —————————————————————————————————————————————————
def main():
    p = argparse.ArgumentParser(description="Improved Genre Classifier")
    p.add_argument('cmd', choices=['init','import','preproc','features','train','predict'])
    p.add_argument('--text', help='Текст для predict')
    p.add_argument('--model', choices=['svm','rf'], default='svm')
    args = p.parse_args()

    if args.cmd=='init':       init_db()
    elif args.cmd=='import':   import_csv()
    elif args.cmd=='preproc':  Preprocessor().run_all()
    elif args.cmd=='features': FeatureBuilder().fit()
    elif args.cmd=='train':    GenreClassifier().train_and_compare()
    elif args.cmd=='predict':
        if not args.text: 
            print("Помилка: вкажіть --text 'текст'")
            sys.exit(1)
        g, p = GenreClassifier().predict(args.text, model=args.model)
        print(f"🎯 Жанр={g}, ймовірність={p:.2f}")
    else: p.print_help()

if __name__=='__main__':
    main()
