import sys
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = {}
        self.threshold = -1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2freq[word] = 0
            self.word2idx[word] = len(self.idx2word) - 1
        self.word2freq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def refactor(self, threshold):
        new_word2idx = {}
        new_idx2word = ['<unk>']
        num_word = 1
        for idx in range(len(self.idx2word)):
            word = self.idx2word[idx]
            if self.word2freq[word] >= threshold:
                new_idx2word.append(word)
                new_word2idx[word] = num_word
                num_word += 1
            else:
                new_word2idx[word] = 0
        self.idx2word = new_idx2word
        self.word2idx = new_word2idx


def get_day(str):
    if str == 'Sun':
        return 6
    if str == 'Sat':
        return 5
    if str == 'Fri':
        return 4
    if str == 'Thu':
        return 3
    if str == 'Wed':
        return 2
    if str == 'Tue':
        return 1
    if str == 'Mon':
        return 0
    #print(str)
    return -1


def get_hour(str):
    a = str.split(':')
    return int(a[0])


def chinese_email_data_set(path, threshold=70000):
    print('[chinese_email_data_set]: start reading in data ...')
    label_path = path + 'label/'
    label_file = open(label_path + 'index_cut', 'r')
    dictionary = Dictionary()
    full_data = []
    x_mailer_dict = Dictionary()
    num_lines = 0
    # load data
    for line in label_file.readlines():
        num_lines += 1
        box = line.split(' ')
        label = (box[0] == 'spam')
        path_file = label_path + box[1][:-1]
        mail_file = open(path_file, 'r')
        tot = 0
        content_flag = False
        word_bag = []
        x_mailer = -1
        day, hour = -1, -1
        # get mail file
        for sub_line in mail_file.readlines():
            tot = tot + 1
            if len(sub_line) == 1:
                content_flag = True
            # x-mailer
            if sub_line[:8] == 'X-Mailer':
                x_mailer = x_mailer_dict.add_word(sub_line[10:])
            # different kind of date
            if sub_line[:5] == 'Date:':
                ss = sub_line.split(' ')
                try:
                    day = get_day(ss[1][:3])
                    hour = get_hour(ss[5])
                except:
                    pass
                    #print(sub_line)
            if sub_line[:6] == 'Date :':
                ss = sub_line.split(' ')
                try:
                    day = get_day(ss[4])
                    hour = int(ss[17])
                except:
                    pass
                    #print(sub_line)
            # external data: omit them
            if sub_line[:4] == 'Date' and sub_line[:5] != 'Date:' and sub_line[:6] != 'Date :':
                pass
                #print(sub_line)
            if content_flag:
                for token in sub_line[:-1].split(' '):
                    if token != ' ':
                        word_bag.append(dictionary.add_word(token))
        # combine label & feature together
        full_data.append((label, (word_bag, x_mailer, day, hour)))
        if num_lines % 1000 == 0:
            print('[chinese_email_data_set]: finish %d files ...' % num_lines)
        if num_lines == threshold:
            break
    label_file.close()

    print('[chinese_email_data_set]: finished, %d items ...' % len(full_data))
    return full_data, dictionary


def shuffle_data(full_data):
    for i in range(len(full_data)):
        j = np.random.randint(i + 1)
        tmp = full_data[i]
        full_data[i] = full_data[j]
        full_data[j] = tmp
    division_n = int(len(full_data) * 0.80)
    train = full_data[0: division_n]
    test = full_data[division_n: len(full_data)]
    return train, test


def get_taobao_reviews(path):
    data_path = path + 'exp2.train.csv'
    data_file = open(data_path, 'r')
    pred_path = path + 'exp2.validation_review.csv'
    pred_file = open(pred_path, 'r')
    full_data = []
    pred_data = []
    num_lines = 0
    for line in data_file.readlines():
        num_lines += 1
        if num_lines == 1:
            continue
        _ = line[:-1].split(',')
        tag = int(_[0])
        data = _[1].split(' ')
        full_data.append((tag, data))
    num_lines = 0
    for line in pred_file.readlines():
        num_lines += 1
        if num_lines == 1:
            continue
        _ = line[:-1].split(',')
        data = _[1].split(' ')
        pred_data.append(data)
    return full_data, pred_data

