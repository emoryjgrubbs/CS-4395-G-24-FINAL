import kagglehub
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from tqdm import tqdm

unk = '<UNK>'

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
    X = data['cleaned_text']
    y = data['generated']
    # for testing
    X, _, y, _ \
        = train_test_split(X, y, test_size=0.99, random_state=1)
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
                print("TRAIN ACCURACY:", accuracy_score(y_train, pred_train))
                print("TRAIN PRECISION:", precision_score(y_train, pred_train))
                print("TRAIN F1:", f1_score(y_train, pred_train))
                print("TRAIN RECALL:", recall_score(y_train, pred_train))
                print("TRAIN CONFUSION MATRIX:", confusion_matrix(y_train, pred_train))
                pred_valid = sm.predict(valid_vec)
                print("VALID ACCURACY:", accuracy_score(y_val, pred_valid))
                print("VALID PRECISION:", precision_score(y_val, pred_valid))
                print("VALID F1:", f1_score(y_val, pred_valid))
                print("VALID RECALL:", recall_score(y_val, pred_valid))
                print("VALID CONFUSION MATRIX:", confusion_matrix(y_val, pred_valid))
                pred_test = sm.predict(test_vec)
                print("TEST ACCURACY:", accuracy_score(y_test, pred_test))
                print("TEST PRECISION:", precision_score(y_test, pred_test))
                print("TEST F1:", f1_score(y_test, pred_test))
                print("TEST RECALL:", recall_score(y_test, pred_test))
                print("TEST CONFUSION MATRIX:", confusion_matrix(y_test, pred_test))
                print()

    print("----- LOGISTIC REGRESSION ------")
    for penalty, solvers in tqdm([(None, ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']), 
                                  ('l2', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
                                  ('l1', ['liblinear', 'saga']),
                                  ('elasticnet', ['saga'])]):
        for solver in solvers:
            cs = [0.001, 0.01, 0.1, 1, 2, 5]
            if penalty == None:
                cs = [1]
            for c in cs:
                lr = linear_model.LogisticRegression(penalty=penalty, solver=solver, C=c, max_iter=1000)
                lr.fit(train_vec, y_train)

                print("-- PENALTY:", penalty, "\tSOLVER:", solver, "\tC:", c, "--")
                pred_train = lr.predict(train_vec)
                print("TRAIN ACCURACY:", accuracy_score(y_train, pred_train))
                print("TRAIN PRECISION:", precision_score(y_train, pred_train))
                print("TRAIN F1:", f1_score(y_train, pred_train))
                print("TRAIN RECALL:", recall_score(y_train, pred_train))
                print("TRAIN CONFUSION MATRIX:", confusion_matrix(y_train, pred_train))
                pred_valid = lr.predict(valid_vec)
                print("VALID ACCURACY:", accuracy_score(y_val, pred_valid))
                print("VALID PRECISION:", precision_score(y_val, pred_valid))
                print("VALID F1:", f1_score(y_val, pred_valid))
                print("VALID RECALL:", recall_score(y_val, pred_valid))
                print("VALID CONFUSION MATRIX:", confusion_matrix(y_val, pred_valid))
                pred_test = lr.predict(test_vec)
                print("TEST ACCURACY:", accuracy_score(y_test, pred_test))
                print("TEST PRECISION:", precision_score(y_test, pred_test))
                print("TEST F1:", f1_score(y_test, pred_test))
                print("TEST RECALL:", recall_score(y_test, pred_test))
                print("TEST CONFUSION MATRIX:", confusion_matrix(y_test, pred_test))
                print()
