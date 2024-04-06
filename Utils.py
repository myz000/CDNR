from sklearn.metrics import roc_auc_score
import random
import numpy as np
# from Utils import NPRATIO

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def evaluate(test_impressions,news_scoring,user_scoring):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        score = np.dot(nvs,uv)

        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
        
    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    
    return AUC, MRR, nDCG5, nDCG10