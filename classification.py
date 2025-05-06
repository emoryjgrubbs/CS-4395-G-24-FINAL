import kagglehub
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from tqdm import tqdm
import csv

unk = '<UNK>'
csv_path = 'output.csv'
csv_fields = ['portion',
              'model',
              'kernel',
              'degree',
              'model',
              'penalty',
              'solver',
              'c',
              'train_acc',
              'train_pre',
              'train_f1',
              'train_recall',
              'train_conf_mat',
              'val_acc',
              'val_pre',
              'val_f1',
              'val_recall',
              'val_conf_mat',
              'test_acc',
              'test_pre',
              'test_f1',
              'test_recall',
              'test_conf_mat'
              ]

# Download latest version & load dataset
path = kagglehub.dataset_download("shanegerami/ai-vs-human-text")
path += "/AI_Human.csv"


# Preprocessing function
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # tokenization
    tokens = text.split()
    return ' '.join(tokens)


def make_vocab(data):
    vocab = set()
    for document in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document in tqdm(data):
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append(vector)
    return vectorized_data


if __name__ == "__main__":
    print("--------- LOADING DATA ---------")
    data = pd.read_csv(path)
    print("------ DATA PREPROCESSING ------")
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    print("-------- SPLITTING DATA --------")
    # for conienience
    X_full = data['cleaned_text']
    y_full = data['generated']
    for portion in [0.2, 0.4, 0.6, 0.8, 1]:
        _, X, _, y \
            = train_test_split(X_full, y_full, test_size=portion, random_state=1)
        # test set split 70 : val/test set split 30
        X_train, X_rem, y_train, y_rem \
            = train_test_split(X, y, test_size=0.3, random_state=1)
        # split the 30% val and test in half
        X_val, X_test, y_val, y_test \
            = train_test_split(X_rem, y_rem, test_size=0.5, random_state=1)
        # form vocab
        vocab = make_vocab(X_train)
        vocab, word2index, index2word = make_indices(vocab)
        # convert data to vectorized representations
        print("------ DATA VECTORIZATION ------")
        train_vec = convert_to_vector_representation(X_train, word2index)
        valid_vec = convert_to_vector_representation(X_val, word2index)
        test_vec = convert_to_vector_representation(X_test, word2index)

        with open(csv_path, 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

            writer.writeheader()

        print("------------- SVM --------------")
        for kernel in tqdm(['linear', 'poly', 'rbf', 'sigmoid']):
            degrees = 2
            if kernel == 'poly':
                degrees = 7
            for degree in range(1, degrees):
                for c in [0.001, 0.01, 0.1, 1, 2, 5]:
                    sm = svm.SVC(kernel=kernel, degree=degree, C=c)
                    sm.fit(train_vec, y_train)

                    print("-- KERNEL:", kernel, "\tDEGREE:", degree, "\tC:", c, "--")
                    pred_train = sm.predict(train_vec)
                    train_acc = accuracy_score(y_train, pred_train)
                    print("TRAIN ACCURACY:", train_acc)
                    train_pre = precision_score(y_train, pred_train)
                    print("TRAIN PRECISION:", train_pre)
                    train_f1 = f1_score(y_train, pred_train)
                    print("TRAIN F1:", train_f1)
                    train_recall = recall_score(y_train, pred_train)
                    print("TRAIN RECALL:", train_recall)
                    train_conf_mat = confusion_matrix(y_train, pred_train)
                    print("TRAIN CONFUSION MATRIX:", train_conf_mat)
                    pred_valid = sm.predict(valid_vec)
                    val_acc = accuracy_score(y_val, pred_valid)
                    print("VALID ACCURACY:", val_acc)
                    val_pre = precision_score(y_val, pred_valid)
                    print("VALID PRECISION:", val_pre)
                    val_f1 = f1_score(y_val, pred_valid)
                    print("VALID F1:", val_f1)
                    val_recall = recall_score(y_val, pred_valid)
                    print("VALID RECALL:", val_recall)
                    val_conf_mat = confusion_matrix(y_val, pred_valid)
                    print("VALID CONFUSION MATRIX:", val_conf_mat)
                    pred_test = sm.predict(test_vec)
                    test_acc = accuracy_score(y_test, pred_test)
                    print("TEST ACCURACY:", test_acc)
                    test_pre = precision_score(y_test, pred_test)
                    print("TEST PRECISION:", test_pre)
                    test_f1 = f1_score(y_test, pred_test)
                    print("TEST F1:", test_f1)
                    test_recall = recall_score(y_test, pred_test)
                    print("TEST RECALL:", test_recall)
                    test_conf_mat = confusion_matrix(y_test, pred_test)
                    print("TEST CONFUSION MATRIX:", test_conf_mat)
                    print()

                    row = {'portion': portion,
                           'model': 'SVM',
                           'kernel': kernel,
                           'degree': degree,
                           'c': c,
                           'train_acc': train_acc,
                           'train_pre': train_pre,
                           'train_f1': train_f1,
                           'train_recall': train_recall,
                           'train_conf_mat': train_conf_mat,
                           'val_acc': val_acc,
                           'val_pre': val_pre,
                           'val_f1': val_f1,
                           'val_recall': val_recall,
                           'val_conf_mat': val_conf_mat,
                           'test_acc': test_acc,
                           'test_pre': test_pre,
                           'test_f1': test_f1,
                           'test_recall': test_recall,
                           'test_conf_mat': test_conf_mat,
                           }

                    with open(csv_path, 'a') as csvfile:
                        # creating a csv dict writer object
                        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

                        writer.writerow(row)

        print("----- LOGISTIC REGRESSION ------")
        for penalty, solvers in tqdm([(None, ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']), 
                                      ('l2', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
                                      ('l1', ['liblinear', 'saga'])]):
            for solver in solvers:
                cs = [0.001, 0.01, 0.1, 1, 2, 5]
                if penalty == None:
                    cs = [1]
                for c in cs:
                    lr = linear_model.LogisticRegression(penalty=penalty, solver=solver, C=c, max_iter=1000)
                    lr.fit(train_vec, y_train)

                    print("-- PENALTY:", penalty, "\tSOLVER:", solver, "\tC:", c, "--")
                    pred_train = lr.predict(train_vec)
                    train_acc = accuracy_score(y_train, pred_train)
                    print("TRAIN ACCURACY:", train_acc)
                    train_pre = precision_score(y_train, pred_train)
                    print("TRAIN PRECISION:", train_pre)
                    train_f1 = f1_score(y_train, pred_train)
                    print("TRAIN F1:", train_f1)
                    train_recall = recall_score(y_train, pred_train)
                    print("TRAIN RECALL:", train_recall)
                    train_conf_mat = confusion_matrix(y_train, pred_train)
                    print("TRAIN CONFUSION MATRIX:", train_conf_mat)
                    pred_valid = lr.predict(valid_vec)
                    val_acc = accuracy_score(y_val, pred_valid)
                    print("VALID ACCURACY:", val_acc)
                    val_pre = precision_score(y_val, pred_valid)
                    print("VALID PRECISION:", val_pre)
                    val_f1 = f1_score(y_val, pred_valid)
                    print("VALID F1:", val_f1)
                    val_recall = recall_score(y_val, pred_valid)
                    print("VALID RECALL:", val_recall)
                    val_conf_mat = confusion_matrix(y_val, pred_valid)
                    print("VALID CONFUSION MATRIX:", val_conf_mat)
                    pred_test = lr.predict(test_vec)
                    test_acc = accuracy_score(y_test, pred_test)
                    print("TEST ACCURACY:", test_acc)
                    test_pre = precision_score(y_test, pred_test)
                    print("TEST PRECISION:", test_pre)
                    test_f1 = f1_score(y_test, pred_test)
                    print("TEST F1:", test_f1)
                    test_recall = recall_score(y_test, pred_test)
                    print("TEST RECALL:", test_recall)
                    test_conf_mat = confusion_matrix(y_test, pred_test)
                    print("TEST CONFUSION MATRIX:", test_conf_mat)
                    print()

                    row = {'portion': portion,
                           'model': 'linear regression',
                           'penalty': penalty,
                           'solver': solver,
                           'c': c,
                           'train_acc': train_acc,
                           'train_pre': train_pre,
                           'train_f1': train_f1,
                           'train_recall': train_recall,
                           'train_conf_mat': train_conf_mat,
                           'val_acc': val_acc,
                           'val_pre': val_pre,
                           'val_f1': val_f1,
                           'val_recall': val_recall,
                           'val_conf_mat': val_conf_mat,
                           'test_acc': test_acc,
                           'test_pre': test_pre,
                           'test_f1': test_f1,
                           'test_recall': test_recall,
                           'test_conf_mat': test_conf_mat,
                           }

                    with open(csv_path, 'a') as csvfile:
                        # creating a csv dict writer object
                        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

                        writer.writerow(row)
