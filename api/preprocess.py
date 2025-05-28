# api/preprocess.py

import spacy

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load('uk_core_news_lg',
                              disable=['parser','ner','textcat'])

    def clean(self, text: str) -> str:
        """Замінюємо переноси рядків на пробіли, обрізаємо зайве."""
        return text.replace('\n', ' ').strip()

    def lemmatize(self, text: str) -> str:
        """Лематизуємо через spaCy, відкидаємо пунктуацію й пробіли."""
        doc = self.nlp(text)
        return ' '.join(
            tok.lemma_
            for tok in doc
            if not tok.is_punct and not tok.is_space
        )
