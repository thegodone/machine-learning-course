import numpy as np

# evaluate the binary classification problem
def binary_classifier(model, test_data):
    tot = [0.0, 0.0]
    correct = [0.0, 0.0]
    num_data = 0
    for data in test_data:
        num_data += 1
        tot[data[0]] += 1
        # get the prediction from model
        pred = model.query(data[1]) >= 0.5
        # conclude some statistics
        if pred == data[0]:
            correct[data[0]] += 1

    # return result
    overall_cnt = tot[0] + tot[1]
    overall_cor = correct[0] + correct[1]
    acc = overall_cor / (overall_cnt + 0.0)

    prec = (correct[1] + 0.0) / (correct[1] + tot[0] - correct[0])
    recl = (correct[1] + 0.0) / tot[1]
    f1 = (prec * recl) * 2 / (prec + recl)

    return acc, prec, recl, f1
