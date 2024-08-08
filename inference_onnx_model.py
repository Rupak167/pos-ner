import os
import re
import numpy as np
import pandas as pd
import onnxruntime as ort
from tensorflow.keras.preprocessing.sequence import pad_sequences
from environs import Env

class NER_POS_Inference:
    def __init__(self, model_path, data_path, max_len):
        self.session = ort.InferenceSession(model_path)
        self.max_len = max_len
        self.data_path = data_path
        self.word2idx, self.pos2idx, self.ner2idx = self.load_dictionaries()
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}
        self.idx2ner = {v: k for k, v in self.ner2idx.items()}

    def predict(self, sentence):
        processed_sentence = self.preprocess_sentence(sentence)
        ort_inputs = {self.session.get_inputs()[0].name: processed_sentence.astype(np.float32)}
        ort_outs = self.session.run(None, ort_inputs)
        
        pos_pred = ort_outs[0]
        ner_pred = ort_outs[1]

        pos_pred_labels = np.argmax(pos_pred, axis=-1).flatten()
        ner_pred_labels = np.argmax(ner_pred, axis=-1).flatten()
        
        pos_tags = [self.idx2pos[idx] for idx in pos_pred_labels if idx in self.idx2pos]
        ner_tags = [self.idx2ner[idx] for idx in ner_pred_labels if idx in self.idx2ner]
        print(f"POS Tags: {pos_tags}")
        print(f"NER Tags: {ner_tags}")
        return pos_tags[:len(sentence.split())], ner_tags[:len(sentence.split())]

    def clean_token(self, token):
        unwanted_chars = r"[()'\",|?]"
        return re.sub(unwanted_chars, '', token)
        
    def preprocess_sentence(self, sentence):
        words = sentence.strip().split()
        words = [self.clean_token(word) for word in words]
        word_indices = [self.word2idx.get(word, 0) for word in words]
        padded_sequence = pad_sequences([word_indices], maxlen=self.max_len, padding='post', value=len(self.word2idx) - 1)
        return padded_sequence

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

if __name__ == "__main__":
    env = Env()
    env.read_env()
    model_path = env.str('ONNX_MODEL_PATH', None)
    data_path = env.str('DATA_PATH', None)
    
    inference = NER_POS_Inference(model_path=model_path, data_path=data_path, max_len=50)
    
    sentence = "আপনার ইনপুট বাক্য এখানে"

    pos_tags, ner_tags = inference.predict(sentence)

    print("POS Tags:", pos_tags)
    print("NER Tags:", ner_tags)
