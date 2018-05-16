import numpy as np
from data import get_taobao_reviews
from data import Dictionary
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from model import Bagging

import logging
import sys


LabeledSentence = gensim.models.doc2vec.LabeledSentence


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def doc2vec_train(x_train, x_test, size=400, epoch_num=10):
    model_dm = gensim.models.Doc2Vec(min_count=1, window=6, vector_size=size,
                                     negative=5, workers=4)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=6, vector_size=size,
                                       negative=5, dm=0, workers=4)
    sentences = x_train + x_test
    model_dm.build_vocab(sentences)
    model_dm.train(sentences, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dbow.build_vocab(sentences)
    model_dbow.train(sentences, total_examples=model_dm.corpus_count, epochs=epoch_num)
    
    model_dm.save('./model_dm.d2v')
    model_dbow.save('./model_dbow.d2v')

    return model_dm, model_dbow


def doc2vec_load(size):
    model_dm = Doc2Vec.load('./model_dm.d2v')
    model_dbow = Doc2Vec.load('./model_dbow.d2v')
    return model_dm, model_dbow


def get_vectors(model_dm, model_dbow, x_train, x_test, size):
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
    return train_vecs, test_vecs


def labelize_reviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def linear_svc(train_vecs, train_label, test_vecs, ground_truth=None, return_pred=False):
    clf = svm.LinearSVC(C=0.01)
    calibrated_svc = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
    calibrated_svc.fit(train_vecs, train_label)

    pred = calibrated_svc.predict(test_vecs)
    pred_prob = calibrated_svc.predict_proba(test_vecs)
    pred_prob_train = calibrated_svc.predict_proba(train_vecs)
    pred_train = pred_prob_train[:, 2] - pred_prob_train[:, 0]

    print('train_rmse = %.4f' % np.sqrt(np.mean((pred_train - train_label) * (pred_train - train_label))))

    if return_pred:
        pred = pred_prob[:, 2] - pred_prob[:, 0]
        print('valid_rmse = %.4f' % np.sqrt(np.mean((pred - ground_truth) * (pred - ground_truth))))
    else:
        output = open('svc_pred.csv', 'w')
        for i in range(len(pred)):
            output.write('%d,%.5f\n' % (i + 1, pred_prob[i][2] - pred_prob[i][0]))


def svc(train_vecs, train_label, test_vecs, return_pred=False):
    clf = svm.SVC(probability=True)
    clf.fit(train_vecs, train_label)
    pred = clf.predict_proba(test_vecs)

    pred_train = clf.predict_proba(train_vecs)
    pred_train = pred_train[:, 2] - pred_train[:, 0]

    print('train_rmse = %.4f' % np.sqrt(np.mean((pred_train - train_label) * (pred_train - train_label))))

    if return_pred:
        return pred[:, 2] - pred[:, 0]
    else:
        output = open('svr_pred.csv', 'w')
        for i in range(len(pred)):
            output.write('%d,%.5f\n' % (i + 1, max(min(pred[i], 1.0), -1)))


def dtree(train_vecs, train_label, test_vecs, return_pred=False):
    clf = DecisionTreeClassifier(random_state=0,max_depth=30,min_samples_split=10,min_samples_leaf=5)
    clf.fit(train_vecs, train_label)
    pred = clf.predict_proba(test_vecs)

    pred_train = clf.predict_proba(train_vecs)
    pred_train = pred_train[:, 2] - pred_train[:, 0]
    print('train_rmse = %.4f' % np.sqrt(np.mean((pred_train - train_label) * (pred_train - train_label))))
    return pred[:, 2] - pred[:, 0]


def bagging(train_vecs, train_label, test_vecs, base, ground_truth=None, return_pred=False):
    if base == 'dtree':
        clf = DecisionTreeClassifier(random_state=0,max_depth=30,min_samples_split=10,min_samples_leaf=5)
    else:
        pclf = svm.LinearSVC(C=0.2)
        clf = CalibratedClassifierCV(pclf, method='sigmoid', cv=3)
    bg = Bagging(clf, 5)
    pred = bg.pred(train_vecs, train_label, len(train_vecs), 
        len(test_vecs) + len(train_vecs), 3, np.concatenate((train_vecs, test_vecs)))
    train_examples = len(train_vecs)
    pred_train = pred[:train_examples, :]
    pred_train = pred_train[:, 2] - pred_train[:, 0]

    print('train_rmse = %.4f' % np.sqrt(np.mean((pred_train - train_label) * (pred_train - train_label))))

    pred_prob = pred[train_examples:, :]
    if return_pred:
        pred = pred_prob[:, 2] - pred_prob[:, 0]
        print('valid_rmse = %.4f' % np.sqrt(np.mean((pred - ground_truth) * (pred - ground_truth))))
    else:
        output = open('bagging_pred.csv', 'w')
        for i in range(len(pred)):
            output.write('%d,%.5f\n' % (i + 1, pred_prob[i][2] - pred_prob[i][0]))


def adaboost(train_vecs, train_label, test_vecs, base, ground_truth=None, return_pred=False):
    if base == 'dtree':
        clf = DecisionTreeClassifier(random_state=0,max_depth=30,min_samples_split=10,min_samples_leaf=5)
    else:
        print('Use svc')
        pclf = svm.LinearSVC(C=1)
        clf = CalibratedClassifierCV(pclf, method='sigmoid', cv=3)
    ada = AdaBoostClassifier(clf, 2, learning_rate=10)
    ada.fit(train_vecs, train_label)
    pred_train = ada.predict_proba(train_vecs)
    pred_train = pred_train[:, 2] - pred_train[:, 0]
    print('train_rmse = %.4f' % np.sqrt(np.mean((pred_train - train_label) * (pred_train - train_label))))

    pred_prob = ada.predict_proba(test_vecs)
    if return_pred:
        pred = pred_prob[:, 2] - pred_prob[:, 0]
        print('valid_rmse = %.4f' % np.sqrt(np.mean((pred - ground_truth) * (pred - ground_truth))))
    else:
        output = open('ada_pred.csv', 'w')
        for i in range(len(pred)):
            output.write('%d,%.5f\n' % (i + 1, pred_prob[i][2] - pred_prob[i][0]))



def split_data(full_data):
    division_n = int(len(full_data) * 0.80)
    train = full_data[0: division_n]
    test = full_data[division_n: len(full_data)]
    return train, test


log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

train, test = get_taobao_reviews('reviews/')
train_sentences = []
train_label = []
for item in train:
    train_sentences.append(item[1])
    train_label.append(item[0])


train_labelized = labelize_reviews(train_sentences, 'TRAIN')
test_labelized = labelize_reviews(test, 'TEST')

size = 400
epoch_num = 30

make_pred = False

if False:
    model_dm, model_dbow = doc2vec_train(train_labelized, test_labelized, size, epoch_num)
else:
    model_dm, model_dbow = doc2vec_load(size)
    train_doc2vecs, test_doc2vecs = get_vectors(model_dm, model_dbow, train_labelized, test_labelized, size)

    dictionary = Dictionary()
    for sentence in train_sentences + test:
        for word in sentence:
            dictionary.add_word(word)
    dictionary.refactor(10)
    print('vocab size = %d' % len(dictionary))
    voc_len = len(dictionary)
    voc_len += 1
    train_vecs = np.zeros((len(train_sentences), voc_len))
    test_vecs = np.zeros((len(test), voc_len))
    for i in range(len(train_sentences)):
        sentence = train_sentences[i]
        for word in sentence:
            train_vecs[i, dictionary.word2idx[word]] += 1
            train_vecs[i, voc_len - 1] += 1
    for i in range(len(test)):
        sentence = test[i]
        for word in sentence:
            test_vecs[i, dictionary.word2idx[word]] += 1
            test_vecs[i, voc_len - 1] += 1
    
    train_vecs = np.concatenate((train_vecs, train_doc2vecs), axis=1)
    test_vecs = np.concatenate((test_vecs, test_doc2vecs), axis=1)
    print('feature extraction finished...')

    if make_pred:
        # pure linear svm
        linear_svc(train_vecs, train_label, test_vecs, return_pred=False)

    else:
        train_vecs, valid_vecs = split_data(train_vecs)
        train_label, valid_label = split_data(train_label)
        ground_truth = np.array(valid_label)
        # pure linear svm
        # pred = linear_svc(train_vecs, train_label, valid_vecs, ground_truth=ground_truth, return_pred=True)
        # ada boost
        bagging(train_vecs, train_label, valid_vecs, 'svc', ground_truth=ground_truth, return_pred=True)
    #pred = bagging(train_vecs, train_label, valid_vecs, 'dtree', return_pred=True)
    #linear_svr(train_vecs, train_label, test_vecs, return_pred=False)
