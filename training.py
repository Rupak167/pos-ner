import re
import os
import pandas as pd
import numpy as np
from environs import Env
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

env = Env()
env.read_env()

class NER_POS_Trainer:
    def __init__(self, data_path, max_len, batch_size, epochs):
        self.data_path = data_path
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        
    def clean_token(self, token):
        unwanted_chars = r"[()'\",|?]"
        return re.sub(unwanted_chars, '', token)

    def load_and_preprocess_data(self):
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
        self.num_words = len(words)

        pos_tags = list(set(data['pos_tag'].values))
        self.num_pos_tags = len(pos_tags)

        ner_tags = list(set(data['ner_tag'].values))
        self.num_ner_tags = len(ner_tags)

        self.word2idx = {w: i+1 for i, w in enumerate(words)}
        self.pos2idx = {p: i for i, p in enumerate(pos_tags)}
        self.ner2idx = {n: i for i, n in enumerate(ner_tags)}

        return data

    def get_sentences(self, data):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['token'].tolist(), s['pos_tag'].tolist(), s['ner_tag'].tolist())]
        grouped = data.groupby("sentence_id").apply(agg_func)
        return [s for s in grouped]

    def prepare_data(self, sentences):
        print("Preparing data...")
        X = [[self.word2idx[w[0]] for w in s] for s in sentences]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding='post', value=self.num_words - 1)

        y_pos = [[self.pos2idx[w[1]] for w in s] for s in sentences]
        y_pos = pad_sequences(maxlen=self.max_len, sequences=y_pos, padding='post')
        y_pos = [to_categorical(i, num_classes=self.num_pos_tags) for i in y_pos]

        y_ner = [[self.ner2idx[w[2]] for w in s] for s in sentences]
        y_ner = pad_sequences(maxlen=self.max_len, sequences=y_ner, padding='post')
        y_ner = [to_categorical(i, num_classes=self.num_ner_tags) for i in y_ner]

        return X, y_pos, y_ner

    def build_model(self):
        print("Building model...")
        input_word = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.num_words, output_dim=self.max_len, input_length=self.max_len)(input_word)
        model = SpatialDropout1D(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

        pos_output = TimeDistributed(Dense(self.num_pos_tags, activation='softmax'), name='pos_output')(model)
        ner_output = TimeDistributed(Dense(self.num_ner_tags, activation='softmax'), name='ner_output')(model)

        self.model = Model(inputs=input_word, outputs=[pos_output, ner_output])
        self.model.compile(optimizer="adam",
                           loss={'pos_output': 'categorical_crossentropy', 'ner_output': 'categorical_crossentropy'},
                           metrics={'pos_output': 'accuracy', 'ner_output': 'accuracy'})

    def train_model(self, X, y_pos, y_ner):
        print("Training model...")
        x_train, x_test, y_pos_train, y_pos_test, y_ner_train, y_ner_test = train_test_split(
            X, y_pos, y_ner, test_size=0.1, random_state=1
        )

        self.model.fit(
            x_train, {'pos_output': np.array(y_pos_train), 'ner_output': np.array(y_ner_train)},
            validation_split=0.2,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )

        self.evaluate_model(x_test, y_pos_test, y_ner_test)

    def evaluate_model(self, x_test, y_pos_test, y_ner_test):
        print("Evaluating model...")
        y_pos_pred, y_ner_pred = self.model.predict(x_test)

        y_pos_test_labels = np.argmax(np.array(y_pos_test), axis=-1).flatten()
        y_pos_pred_labels = np.argmax(y_pos_pred, axis=-1).flatten()
        y_ner_test_labels = np.argmax(np.array(y_ner_test), axis=-1).flatten()
        y_ner_pred_labels = np.argmax(y_ner_pred, axis=-1).flatten()

        mask = (x_test.flatten() != self.num_words - 1)

        pos_classification_report = classification_report(
            y_pos_test_labels[mask], y_pos_pred_labels[mask], target_names=[k for k, v in self.pos2idx.items()]
        )
        print("POS Tagging Classification Report:\n", pos_classification_report)

        ner_classification_report = classification_report(
            y_ner_test_labels[mask], y_ner_pred_labels[mask], target_names=[k for k, v in self.ner2idx.items()],
            labels=list(self.ner2idx.values())
        )
        print("NER Tagging Classification Report:\n", ner_classification_report)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def run(self, model_path):
        data = self.load_and_preprocess_data()
        sentences = self.get_sentences(data)
        X, y_pos, y_ner = self.prepare_data(sentences)
        self.build_model()
        self.train_model(X, y_pos, y_ner)
        self.save_model(model_path)


if __name__ == "__main__":
    model_path = env.str('MODEL_PATH', None)
    data_path = env.str('DATA_PATH', None)
    trainer = NER_POS_Trainer(data_path= data_path, max_len=50, batch_size=64, epochs=20)
    trainer.run(model_path=model_path)
