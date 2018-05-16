from data import chinese_email_data_set, shuffle_data
from model import NaiveBayes
from evaluation import binary_classifier
import numpy as np

if __name__ == '__main__':
    np.random.seed(1234)
    full_data, dictionary = chinese_email_data_set('trec06c-utf8/')

    accs = []
    precs = []
    recls = []
    f1s = []
    for i in range(5):
        # use different seed
        np.random.seed(1234 + i)
        # get data
        train, test = shuffle_data(full_data)
        # cut data
        n_samples = int(len(train))
        # learn
        model = NaiveBayes(train[:n_samples], dictionary)
        # evaluation
        acc, prec, recl, f1 = binary_classifier(model, test)
        print('Round #%d: %.5f %.5f %.5f %.5f' % (i + 1, acc, prec, recl, f1))
        accs.append(acc)
        precs.append(prec)
        recls.append(recl)
        f1s.append(f1)
    print('[Result]:\n')
    print('accuracy %.5f +/- %.5f' % (np.mean(accs), np.std(accs)))
    print('precision %.5f +/- %.5f' % (np.mean(precs), np.std(precs)))
    print('recall %.5f +/- %.5f' % (np.mean(recls), np.std(recls)))
    print('f1 score %.5f +/- %.5f' % (np.mean(f1s), np.std(f1s)))

