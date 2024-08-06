import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from domain.models import Result

class NER_POS_Prediction:
    def __init__(self, model_path, data_path, max_len):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        self.max_len = max_len
        self.data_path = data_path
        self.word2idx, self.pos2idx, self.ner2idx = self.load_dictionaries()
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}
        self.idx2ner = {v: k for k, v in self.ner2idx.items()}

    def clean_token(self, token):
        unwanted_chars = r"[()'\",|?]"
        return re.sub(unwanted_chars, '', token)
        
    def preprocess_words(self, words):
        word_indices = [self.word2idx.get(word, 0) for word in words]
        padded_sequence = pad_sequences([word_indices], maxlen=self.max_len, padding='post', value=len(self.word2idx) - 1)
        return padded_sequence
    
    def predict(self, sentence):
        words = sentence.strip().split()
        words = [self.clean_token(word) for word in words]

        processed_words = self.preprocess_words(words)
        pos_pred, ner_pred = self.model.predict(processed_words)
        
        pos_pred_labels = np.argmax(pos_pred, axis=-1).flatten()
        ner_pred_labels = np.argmax(ner_pred, axis=-1).flatten()
        
        pos_tags = [self.idx2pos[idx] for idx in pos_pred_labels if idx in self.idx2pos]
        ner_tags = [self.idx2ner[idx] for idx in ner_pred_labels if idx in self.idx2ner]
        logger.info(f"POS Tags: {pos_tags}")
        logger.info(f"NER Tags: {ner_tags}")
        result = []
        for word, pos, ner in zip(words, pos_tags[:len(sentence.split())], ner_tags[:len(sentence.split())]):
            result.append(
                Result(
                    token=word,
                    pos_tag=pos,
                    ner_tag=ner
                )
            )
        return result

    def load_dictionaries(self):
        column_names = ['token', 'pos_tag', 'ner_tag', 'sentence_id']
        data = pd.read_csv(self.data_path, delimiter='\t', quoting=3, names=column_names, encoding='utf-8')
        
        data['sentence_id'] = (data['token'].isnull().cumsum())
        data['token'] = data['token'].str.strip()
        data['pos_tag'] = data['pos_tag'].str.strip()
        data['ner_tag'] = data['ner_tag'].str.strip()
        data = data.dropna().reset_index(drop=True)
        data['token'] = data['token'].apply(self.clean_token)

        words = list(set(data['token'].values))
        words.append("ENDPAD")
        pos_tags = list(set(data['pos_tag'].values))
        ner_tags = list(set(data['ner_tag'].values))

        word2idx = {w: i + 1 for i, w in enumerate(words)}
        pos2idx = {p: i for i, p in enumerate(pos_tags)}
        ner2idx = {n: i for i, n in enumerate(ner_tags)}

        return word2idx, pos2idx, ner2idx