import time
import pandas as pd
import numpy as np
import datetime
import os
import sys

"""
Reproduce Figure 1 in the paper analyzing the MIMIC dataset
"""

DATA_PATH = "drive/MyDrive/mimic_temp_data/"

def read_csv(filepath=DATA_PATH):

    '''
    Read the data
    '''
    DIAGNOSES_ICD = pd.read_csv(filepath + 'DIAGNOSES_ICD.csv')
    NOTES = pd.read_csv(filepath + 'notes_cleaned.csv', engine='c')

    return DIAGNOSES_ICD, NOTES

def event_count_metrics(diagnoses):

    '''
    count number of icd9 codes overall
    '''
    total_icd9_count = diagnoses.groupby(['ICD9_CODE']).size().reset_index(name='counts')
    total_icd9_count = total_icd9_count.sort_values('counts', ascending=False)
    #print(total_icd9_count.head(20))
    total_icd9_count.head(10).plot(x='ICD9_CODE', y='counts')

    total_admissions_diagnosis = diagnoses.groupby(['HADM_ID']).size().reset_index(name='counts')
    total_admissions_diagnosis = total_admissions_diagnosis.sort_values('counts', ascending=True)
    total_admissions_diagnosis= total_admissions_diagnosis.groupby(['counts']).size().reset_index(name='admissions')
    total_admissions_diagnosis = total_admissions_diagnosis[['counts', 'admissions']].drop_duplicates()
    #print(total_admissions_diagnosis.head(40))
    total_admissions_diagnosis.plot(x='counts', y='admissions')

    return total_icd9_count, total_admissions_diagnosis

diagnoses, notes = read_csv(DATA_PATH)
total_icd9_count, total_admissions_diagnosis = event_count_metrics(diagnoses)

# use only the top 20 codes for model training for now as there are 7000 codes total
# also, use only 1 assigned code per visit instead of N
NUM_CLASSES = 5
diagnoses = diagnoses.loc[diagnoses['SEQ_NUM'] == 1]
diagnoses = diagnoses.loc[diagnoses['ICD9_CODE'].isin(total_icd9_count.head(NUM_CLASSES)['ICD9_CODE'])]
#diagnoses = diagnoses.groupby(['SUBJECT_ID', 'HADM_ID']).ICD9_CODE.apply(list).reset_index()
#print(diagnoses)
notes = pd.merge(notes, diagnoses,  how='left', left_on=['SUBJECT_ID','HADM_ID'], right_on = ['SUBJECT_ID','HADM_ID'])
notes = notes[notes['ICD9_CODE'].notnull()]
print(notes.head(2))
print(len(notes.index))

"""
Use word2vec to create vectors of similar words
"""
import os
RANDOM_SEED = 23432098
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

VEC_SIZE = 100
MIN_COUNT = 1
WORKERS=4
import gensim
from gensim.models import Word2Vec
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

train, test = train_test_split(notes, test_size=0.2)

def generate_grams(dataset):
    corpus, grams = [], []
    for note in dataset:
        words = note.split()
        grams = [" ".join(words[i:i+1]) for i in range(0, len(words), 1)]
        corpus.append(grams)

    bigrams_detector = gensim.models.phrases.Phrases(
        corpus, 
        delimiter=" ".encode(), 
        min_count=5, 
        threshold=10
        )
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(
        bigrams_detector[corpus], 
        delimiter=" ".encode(), 
        min_count=5, 
        threshold=10
        )
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    corpus = list(bigrams_detector[corpus])
    corpus = list(trigrams_detector[corpus])
    return corpus

X_train = generate_grams(train['diagnosis_note'])
print(len(X_train))

w2v_model = Word2Vec(sentences=X_train, seed=RANDOM_SEED, size=VEC_SIZE, window=5, min_count=MIN_COUNT, workers=WORKERS)
w2v_model.save("word2vec.model")

def sequence_editing(dataset):
    ## tokenize text
    tokenizer = kprocessing.text.Tokenizer(
        lower=True, 
        split=' ', 
        oov_token="NaN", 
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
    tokenizer.fit_on_texts(dataset)
    dic_vocabulary = tokenizer.word_index

    lst_text2seq= tokenizer.texts_to_sequences(dataset)

    sequence = kprocessing.sequence.pad_sequences(
        lst_text2seq, 
        maxlen=len(X_train), 
        padding="post", 
        truncating="post"
        )
    return sequence, dic_vocabulary


X_test = generate_grams(test)
X_test, dic_vocabulary = sequence_editing(X_test)
X_train, dic_vocabulary = sequence_editing(X_train)

embeddings = np.zeros((len(dic_vocabulary)+1, VEC_SIZE))

for word, idx in dic_vocabulary.items():
    try:
        embeddings[idx] =  w2v_model[word]
    except:
        pass
"""
Model creation
"""
import tensorflow as tf
## code attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

## input
x_in = layers.Input(shape=(len(X_train),))
## embedding
x = layers.Embedding(
    input_dim=embeddings.shape[0],  
    output_dim=embeddings.shape[1], 
    weights=[embeddings],
    input_length=len(X_train), 
    trainable=False
    )(x_in)
## apply attention
x = attention_layer(x, neurons=len(X_train))
## 2 layers of bidirectional lstm
x = layers.Bidirectional(
    layers.LSTM(
        units=NUM_CLASSES, 
        dropout=0.2, 
        return_sequences=True
        )
    )(x)
x = layers.Bidirectional(
    layers.LSTM(
        units=NUM_CLASSES, 
        dropout=0.2
        )
    )(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
## compile
model = models.Model(x_in, y_out)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
    metrics=['accuracy']
    )

model.summary()
"""
Alternative test using CNN instead results in readme

from tensorflow.keras import Sequential, models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

textcnnmodel = Sequential()
textcnnmodel.add((layers.Input(shape=(len(X_train),))))
textcnnmodel.add(
    layers.Embedding(
        input_dim=embeddings.shape[0],  
        output_dim=embeddings.shape[1], 
        weights=[embeddings],
        input_length=len(X_train), 
        trainable=False
    )
)
textcnnmodel.add(layers.Conv1D(128, 5, activation='relu'))
textcnnmodel.add(layers.GlobalMaxPooling1D())
textcnnmodel.add(layers.Dense(5, activation='relu'))
#textcnnmodel.add(layers.Dense(1, activation='sigmoid'))

textcnnmodel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
textcnnmodel.summary() 
"""

"""
Model Training
"""
y_train = train['ICD9_CODE'].values
y_test = test['ICD9_CODE'].values

def get_encoding(dataset):
    checked_codes = {}
    encoded_y_train = []
    unique_counter = 0
    for code in dataset:
        if code not in checked_codes:
            checked_codes[code] = unique_counter
            encoded_y_train.append(unique_counter)
            unique_counter +=1
        else:
            encoded_y_train.append(checked_codes[code])
    return encoded_y_train
y_train = np.array(get_encoding(y_train))
y_test = np.array(get_encoding(y_test))

def train_model(model):
    training = model.fit(
        x=X_train, 
        y=y_train, 
        batch_size=256, 
        epochs=10, 
        shuffle=True, 
        verbose=2, 
        validation_split=0.3,
    )
# use memory profiler to check memory use
%memit train_model(model)

"""
Visualize results
"""
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(training, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(training, 'loss')
plt.ylim(0, None)
