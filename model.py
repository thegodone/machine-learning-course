import numpy as np
from data import Dictionary
import random
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Naive Bayes model for spam e-mail detection
class NaiveBayes(object):
    def __init__(self, train, dictionary, alpha=1e-100, advanced_fea=True):
        print('NaiveBayes for spam detection: start estimating parameters ...')
        tot = np.zeros(2)
        time_tot = np.zeros(2)
        token_tot = np.zeros((2, len(dictionary)))
        mail_tot = np.zeros((2, 158))
        day_tot = np.zeros((2, 7))
        hour_tot = np.zeros((2, 24))
        for item in train:
            tot[item[0]] += 1
            for j in set(item[1][0]):
                token_tot[item[0], j] += 1
            mail_tot[item[0], item[1][1]] += 1
            if item[1][2] >= 0:
                day_tot[item[0], item[1][2]] += 1
                hour_tot[item[0], item[1][3]] += 1
                time_tot[item[0]] += 1
        self.ntoken = len(dictionary)
        self.ndocu = len(train)
        # the positve / negative prob
        self.p_sm = tot[1] / np.sum(tot)
        self.p_hm = tot[0] / np.sum(tot)
        # the cond prob of bag of words
        self.p_x1_sm = (token_tot[1, :] + alpha) / (tot[1] + 2 * alpha)
        self.p_x1_hm = (token_tot[0, :] + alpha) / (tot[0] + 2 * alpha)
        self.p_x0_sm = (tot[1] - token_tot[1, :] + alpha) / (tot[1] + 2 * alpha)
        self.p_x0_hm = (tot[0] - token_tot[0, :] + alpha) / (tot[0] + 2 * alpha)
        # the cond prob of x-mailer
        self.p_ml_sm = (mail_tot[1, :] + alpha) / (tot[1] + 158 * alpha)
        self.p_ml_hm = (mail_tot[0, :] + alpha) / (tot[0] + 158 * alpha)
        # the cond prob of day
        self.p_day_sm = (day_tot[1, :] + alpha) / (time_tot[1] + 7 * alpha)
        self.p_day_hm = (day_tot[0, :] + alpha) / (time_tot[0] + 7 * alpha)
        # the cond prob of hour
        self.p_hour_sm = (hour_tot[1, :] + alpha) / (time_tot[1] + 24 * alpha)
        self.p_hour_hm = (hour_tot[0, :] + alpha) / (time_tot[0] + 24 * alpha)
        self.advanced_fea = advanced_fea
        print('NaiveBayes for spam detection: finished ...')

    def _query(self, token_set, x_mailer, day, hour):
        ident = np.zeros(self.ntoken)
        for i in token_set:
            ident[i] = 1
        # calculate log of prob in order to avoid numerical problem
        log_p_x_sm = np.log(self.p_sm)
        log_p_x_sm += np.sum(np.log(self.p_x1_sm) * ident + \
                             (1 - ident) * np.log(self.p_x0_sm))
        log_p_x_hm = np.log(self.p_hm)
        log_p_x_hm += np.sum(np.log(self.p_x1_hm) * ident + \
                             (1 - ident) * np.log(self.p_x0_hm))
        # advanced features
        if self.advanced_fea:
            # x-mailer
            log_p_x_hm += np.log(self.p_ml_hm[x_mailer])
            log_p_x_sm += np.log(self.p_ml_sm[x_mailer])
            # day
            if day != -1:
                log_p_x_sm += np.log(self.p_day_sm[day])
                log_p_x_hm += np.log(self.p_day_hm[day])
            # hour
            if hour != -1:
                log_p_x_sm += np.log(self.p_hour_sm[hour])
                log_p_x_hm += np.log(self.p_hour_hm[hour])

        delta = log_p_x_sm - log_p_x_hm
        # cut to avoid numerical problem
        delta = min(max(delta, -100), 100)
        sm_hm = np.exp(delta)
        return 1 / (1 + 1 / sm_hm)

    def query(self, data):
        return self._query(data[0], data[1], data[2], data[3])


class Bagging(object):
    def __init__(self, base_estimator, num_votes):
        self.num_votes = num_votes
        self.base_estimator = base_estimator

    def pred(self, X, y, num_data, num_test, num_cate, X_pred):
        final_pred = np.zeros((num_test, num_cate))
        for i in range(self.num_votes): 
            current_X = []
            current_y = []
            for j in range(num_data):
                idx = random.randint(0, num_data - 1)
                current_X.append(X[idx])
                current_y.append(y[idx])
            self.base_estimator.fit(current_X, current_y)
            final_pred += self.base_estimator.predict_proba(X_pred)
        return final_pred / (self.num_votes + 0.0)

def AdaBoost(X, y, base_estimator, num_votes, num_data, num_test, num_cate, X_pred):
    sample_weight = 1.0 / num_data * np.ones((num_data,))
    weights = []
    predicts = []
    for t in range(num_votes):
        # estimator
        pclf = svm.LinearSVC(C=0.01)
        clf = CalibratedClassifierCV(pclf, method='sigmoid', cv=3)

        # fit
        clf.fit(X, y, sample_weight)
        error = clf.predict(X) != y
        error_rate = np.sum(sample_weight * error)

        print('Error rate = %.2f\n' % error_rate)
        if error_rate > 0.5:
            num_votes = t
            break

        beta = error_rate / (1 - error_rate)
        sample_weight = sample_weight * ((1 - error) * beta + error)
        sample_weight /= np.sum(sample_weight)

        #weights.append(float(1 / beta))
        weights.append(float(np.log(1 / beta)))
        print('Final Prediction')
        predicts.append(clf.predict_proba(np.concatenate((X, X_pred))))

        print('Solve %d votes' % (t + 1));

    weights_sum = sum(weights)
    final_pred = np.zeros((num_data + num_test, num_cate))
    for t in range(num_votes):
        final_pred += predicts[t] * weights[t] / weights_sum
    #final_pred = np.exp(final_pred)
    #final_pred = final_pred / np.sum(final_pred, axis=1, keepdims=True)
    #print(final_pred)
    return final_pred



