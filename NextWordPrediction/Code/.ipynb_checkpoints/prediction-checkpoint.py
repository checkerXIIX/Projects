from turtle import back
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os
import string
import csv
import re
import matplotlib.pyplot as plt

from time import time
import random

from tensorflow import keras
from keras.utils.vis_utils import plot_model

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

import io

import time

def get_data(filename):
    lines = []
    f = io.open(filename, mode="r", encoding="utf-8")
    lines = f.read()

    #data = ""
    #for i in lines:
    #    data = ' '.join(lines) #Eingelesene Zeilen werden zu einem großen String

    print(lines[500:1000])
    return lines



def preprocess_data(data):
    print("preprocessing data")
    # Tabs und Zeilenumbrüche + Anführungszeichen entfernen
    data = data.replace('\n', ' ').replace('\r', ' ').replace('\ufeff', '').replace('	“', '').replace('„','')

    #Großbuchstaben entfernen
    data = data.lower()

    # alle Zahlen entfernen
    data = re.sub(r'[0-9]+', '', data)

    # Punkte und Sonderzeichen entfernen
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    data = data.translate(translator)

    #data = ' '.join(filter(str.isalnum, data))

    # einzelne buchstaben entfernen
    # bin ich mir nicht sicher ob sinnvoll, da es bei Bauteilen schon häufig vorkommt und dort sinnvoll ist
    data = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', data)

    #doppelte Leerzeichen entfernen
    data = ' '.join(data.split())

    print(data[:1000])

    return data

def remove_stopwords(data):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(data)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = ""
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence += ' ' + w # nur wörter hinzufügen, die keine Stopwords sind

    filtered_sentence = ' '.join(filtered_sentence.split()) #leerzeichen am anfang entfernen
    print(filtered_sentence[:1000])
    return filtered_sentence


def tokenize_data(data, foldername):
    print("tokenize data")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])

    # saving the tokenizer for predict function.
    # Tokenizer am besten immer neu mit Datum abspeichern
    pickle.dump(tokenizer, open('goethe_without_stopwords.pkl', 'wb'))

    sequence_data = tokenizer.texts_to_sequences([data])[0]
    print("sequence data len", len(sequence_data))
    print(sequence_data[:100])

    vocab_size = len(tokenizer.word_index) + 1
    print('vocab size', vocab_size)
    return sequence_data, vocab_size, len(sequence_data)

def create_sequences(sequence_data, sequence_len, vocab_size, visualization_status):
    sequences = []
    for i in range(sequence_len, len(sequence_data)): # loop beginnt erst ab 5 (sequenzlänge), damit sequenzen immer voll sind
        words = sequence_data[i - sequence_len:i + 1]  # words immer 6 wörter lang (5 input, 1 prediction)
        sequences.append(words)
    print("The Length of sequences are: ", len(sequences))
    print(sequences[:10])
    random.shuffle(sequences)
    print(sequences[:10])
    sequences = np.array(sequences)

    X = []
    y = []
    for i in sequences:
        X.append(i[0:sequence_len])  # input sind die letzten fünf wörter
        y.append(i[sequence_len])  # prediction soll das nächste wort sein

    X = np.array(X)
    y = np.array(y)
    print("The Data is: ", X[:100])
    print("The responses are: ", y[:100])

    if visualization_status == False: ## falls predictions visualisiert werden sollen, die responses nicht one hot encoden
        y = to_categorical(y, num_classes=vocab_size, dtype='uint8')
        print(y[:10])
    return X, y


def build_model_double_lstm(nodes, embedding_size, sequence_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_len))
    model.add(LSTM(nodes, return_sequences=True))
    model.add(LSTM(nodes))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    return model

def build_model_bidirectional_lstm(nodes, embedding_size, sequence_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_len))
    model.add(Bidirectional(LSTM(nodes)))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

def train_model(model, X, y, epochen, batch_size, logdir, model_filename):
    model.summary()
    checkpoint = ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
    tensorboard_Visualization = TensorBoard(log_dir=logdir)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.fit(X, y, epochs=epochen, batch_size=batch_size, validation_split=0.1, callbacks=[checkpoint, reduce, tensorboard_Visualization])


def test_model(X, y, logdir, model_filename):
    model = keras.models.load_model(model_filename)
    tensorboard_Visualization = TensorBoard(log_dir=logdir)
    scores = model.evaluate(X, y, verbose=1, callbacks=[tensorboard_Visualization])
    print(model_filename)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    return scores


def visualization_of_model(X, y, model_filename, save_name):
    print('visalusation started')
    model = keras.models.load_model(model_filename)
    tokenizer = pickle.load(open('goethe_without_stopwords.pkl', 'rb'))

    input_words_list = []
    output_word_list = []
    true_next_word_list = []
    correct_list = []
    index_list = []
    prediction_list = []

    tokens = dict((v, k) for k, v in tokenizer.word_index.items())

    for x in range(len(X)):  # brauch ich die loop ?? oder lässt es sich auch anders lösen?? direkt aus X ein pandas dings machen??
        sequence = pad_sequences([X[x]], maxlen=10, padding='pre')
        #predicted = np.argmax(model.predict(sequence))

        #get three best predictions
        n = 1
        predicted = model.predict(sequence)
        top_3_predictions = []
        predicted_copy = predicted.copy()
        for t in predicted_copy:
            t.sort()
            top_3_predictions.append(t[-1:])
        pred = np.argmax(model.predict(sequence))
        output_word = tokens[pred]

        input = ""
        for i in X[x]:
            input += ' ' + tokens[i]

        true_next_word = tokens[y[x]]

        if output_word == true_next_word:
            correct = True
        else:
            correct = False

        input = ' '.join(input.split())
        input_words_list.append(input)
        output_word_list.append(output_word)
        prediction_list.append(top_3_predictions[0])
        true_next_word_list.append(true_next_word)
        correct_list.append(correct)

    df = pd.DataFrame(list(zip(input_words_list, true_next_word_list, output_word_list, prediction_list, correct_list)), columns=['Sequencen', 'True next word', 'Predicted word', 'Predictions', 'Correct'])
    df_true = df[df['Correct'] == True]

    df.to_csv('models_results_' + save_name + '.csv')
    df_true.to_csv('models_results_true_' + save_name + '.csv')

    display(df)
    display(df_true)
    
def data_generator(sequence_data, sequence_len, vocab_size, batch_size):
    num_samples = len(sequence_data)
    num_batches = int(num_samples / batch_size)
    if num_samples % batch_size:
        num_samples += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch, y_batch = create_sequences(sequence_data[start_idx:end_idx], sequence_len, vocab_size, False)
            yield x_batch, y_batch


def train_model_on_whole_data(epochen, sequence_data, sequence_len, vocab_size, entrys, logdir, model_filename):
    batch_size = 64

    idx_trainingsdata = entrys * 0.9
    idx_validationdata = entrys
    my_training_batch_generator = data_generator(sequence_data[:idx_trainingsdata], sequence_len, vocab_size, batch_size)
    my_validation_batch_generator = data_generator(sequence_data[idx_trainingsdata:idx_validationdata], sequence_len, vocab_size, batch_size)

    steps_per_epoch = len(sequence_data[:idx_trainingsdata]) / batch_size
    validation_steps = len(sequence_data[idx_trainingsdata:idx_validationdata]) / batch_size

    model = build_model_bidirectional_lstm(150, 100, vocab_size)

    checkpoint = ModelCheckpoint(model_filename, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
    tensorboard_Visualization = TensorBoard(log_dir=logdir)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])

    model.fit(x=my_training_batch_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochen,
                verbose=1,
                validation_data = my_validation_batch_generator,
                validation_steps=validation_steps,
                callbacks=[checkpoint, reduce, tensorboard_Visualization])


def trainings_pipeline():
    foldername = "well_withSW_nodes128_test2"
    model1_name = "bidirectional_lstm_150_em100"
    model2_name = "doouble_lstm_150_em500"
    model3_name = "bidirectional_lstm_150_em500"
    file = "1661-0.txt"

    data = get_data(file)
    data = preprocess_data(data)
    data = remove_stopwords(data)
    sequence_data, vocab_size, entrys = tokenize_data(data, foldername)

    sequence_len = 10   # am besten so groß wie möglich, wird aber von Hardware eingegrenzt -> kompromiss finden
    idx_trainingsdata = int(entrys * 0.95)
    print('trainingsdata')
    print(idx_trainingsdata)
    idx_testdata = int(entrys)
    print('testdata')
    print(idx_testdata)
    X, y = create_sequences(sequence_data[:idx_trainingsdata], sequence_len, vocab_size, False)

    embedding_size = 500
    nodes = 256

    #modelle mit unterschiedlicher embedding_size
    #model1 = build_model_bidirectional_lstm(nodes, 100, sequence_len, vocab_size)
    model2 = build_model_double_lstm(nodes, 500, sequence_len, vocab_size)
    model3 = build_model_bidirectional_lstm(nodes, 500, sequence_len, vocab_size)

    batch_size = 64
    #trainiert modelle auf verschiedene anzahl an epochen
    for i in range(8, 12, 4): #i ist die anzahl der epochen
        #train_model(model1, X, y, i, batch_size, 'logs_' + foldername + '/logs_' + model1_name + str(i), foldername + "/" + model1_name + str(i) + ".h5")
        #train_model(model2, X, y, i, batch_size, 'logs_' + foldername + '/logs_' + model2_name + str(i), foldername + "/" + model2_name + str(i) + ".h5")
        #start_time = time.time()
        #train_model(model2, X, y, i, batch_size, 'logs_' + foldername + '/logs_' + model2_name + str(i), foldername + "/" + model2_name + str(i) + ".h5")
        #print("double_lstm_time")
        #print(time.time() - start_time, "seconds")
        start_time = time.time()
        train_model(model3, X, y, i, batch_size, 'logs_' + foldername + '/logs_' + model3_name + str(i), foldername + "/" + model3_name + str(i) + ".h5")
        print("bidirectional_time")
        print(time.time() - start_time, "seconds")


    X_test, y_test = create_sequences(sequence_data[idx_trainingsdata:idx_testdata], sequence_len, vocab_size, False)
    for i in range(8, 12, 4):
        #test_model(X_test, y_test, 'logs_' + foldername + '_evaluation/logs_' + model1_name + str(i), foldername + "/" + model1_name + str(i) + ".h5")
        #test_model(X_test, y_test, 'logs_' + foldername + '_evaluation/logs_' + model2_name + str(i), foldername + "/" + model2_name + str(i) + ".h5")
        #test_model(X_test, y_test, 'logs_' + foldername + '_evaluation/logs_' + model2_name + str(i), foldername + "/" + model2_name + str(i) + ".h5")
        test_model(X_test, y_test, 'logs_' + foldername + '_evaluation/logs_' + model3_name + str(i), foldername + "/" + model3_name + str(i) + ".h5")

    X_test, y_test = create_sequences(sequence_data[idx_trainingsdata-2000:idx_trainingsdata-1000], sequence_len, vocab_size, True)
    for i in range(8, 12, 4):
        #visualization_of_model(X_test, y_test, foldername + "/" + model1_name + str(i) + ".h5", foldername + '_' + model1_name + str(i))
        #visualization_of_model(X_test, y_test, foldername + "/" + model2_name + str(i) + ".h5", foldername + '_' + model2_name + str(i))
        #visualization_of_model(X_test, y_test, foldername + "/" + model2_name + str(i) + ".h5", foldername + '_' + model2_name + str(i))
        visualization_of_model(X_test, y_test, foldername + "/" + model3_name + str(i) + ".h5", foldername + '_' + model3_name + str(i))

def training_on_whole_data():
    file = "data/data/HUBITUS_ISSUES_TASKS_V2_4_CLEANED_GE.csv"
    last_entry = 19905
    data = get_data(file, 0, last_entry)
    data = preprocess_data(data)
    data = remove_stopwords(data)
    sequence_data, vocab_size, entrys = tokenize_data(data)

    foldername = "test7_whole_data_without_stopwords"
    model_name = "bidirectional_lstm_150_"
    logdir = 'logs_' + foldername + '/logs_' + model_name + '20'
    modelfilename = foldername + "/" + model_name + "20.h5"

    sequence_len = 5
    epochen = 10   #am besten zwischen 10 und 15
    train_model_on_whole_data(epochen, sequence_data, sequence_len, vocab_size, entrys, logdir, modelfilename)

    X_test, y_test = create_sequences(sequence_data[800000:850000], sequence_len, vocab_size, True)
    visualization_of_model(X_test, y_test, foldername + "/" + model_name + "10.h5", foldername + '_' + model_name + '10')

# nach anderen evaluierungsmöglichkeiten suchen außer accuracy
# modelle auf unterschiedliche epochen trainieren
# je größer modell --> mehr epochen
# oder  alle auf 10 epochen dann 20 epochen, .....
# einmal alles testen mit stopwords und einmal ohne stopwords


#46875 / 3000000[..............................] - ETA: 396:56: 59 - loss: 6.3870 - accuracy: 0.2274



#### hilfreiche test methoden
import pandas as pd
def print_dataframes():
    csv_file = "models_results_predicitons_borderbidirectional_lstm_150_15.csv"
    #csv_file = "models_results_true_predicitons_borderbidirectional_lstm_150_15.csv"

    csvdatei = open(csv_file, encoding="utf8")

    csv_reader_object = csv.reader(csvdatei)

    false_counter = 0
    true_counter = 0
    zeilennummer = 0
    for row in csv_reader_object:
        if zeilennummer == 0:
            print(f'Spaltennamen sind: {", ".join(row)}')
        else:
            prob = float(row[4][1:-1])
            if(prob >= 0.92):
                if(row[5] == 'False'):
                    false_counter+=1
                elif (row[5] == 'True'):
                    true_counter+=1
        zeilennummer += 1

    print(f'Anzahl Datensätze: {zeilennummer - 1}')
    print('True counter ', true_counter)
    print('False counter ', false_counter)


def compare_tokenizer():
    tokenizer1 = pickle.load(open('tokenizertest1.pkl', 'rb'))
    tokenizer2 = pickle.load(open('tokenizertest2.pkl', 'rb'))

    tokens1 = dict((v, k) for k, v in tokenizer1.word_index.items())
    tokens2 = dict((v, k) for k, v in tokenizer2.word_index.items())

    for i in range(1, len(tokenizer1.word_index.items())):
        if tokens1[i] == tokens2[i]:
            print('True')
        else:
            print('false')

trainings_pipeline()