import numpy as np
import json
from Hypers import *
from Utils import *
from nltk.tokenize import word_tokenize
import os

def read_news(news_paths):
    
    news={}

    news_index={}
    index=1

    word_dict={}
    word_index=1
    
    word_dict['[CLS]'] = word_index
    word_index+=1
    word_dict['[SEP]'] = word_index
    word_index+=1    
    
    
    content_count = {}
    content_dict = {}
    content_index = 1

    entity_dict = {}
    entity_index = 1

    category_dict={}
    category_index = 1

    subcategory_dict={}
    subcategory_index = 1

    for path in news_paths:

        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip('\n').split('\t')
            doc_id,vert,subvert,title,abstract,url,entity,_= splited
            entity = json.loads(entity)
            if doc_id in news_index:
                continue

            news_index[doc_id]=index
            index+=1

            title = word_tokenize(title.lower())  
            abstract = abstract.lower().split()[:MAX_CONTENT]
            entity = [e['WikidataId'] for e in entity]

            news[doc_id]=[vert,subvert,title,entity,abstract]

            for word in title:
                if not(word in word_dict):
                    word_dict[word]=word_index
                    word_index+=1
                    
            for word in abstract:
                if not (word in content_count):
                    content_count[word] = 0
                content_count[word] += 1

            for e in entity:
                if not (e in entity_dict):
                    entity_dict[e] = entity_index
                    entity_index += 1

            if not vert in category_dict:
                category_dict[vert] = category_index
                category_index += 1

            if not subvert in subcategory_dict:
                subcategory_dict[subvert] = subcategory_index
                subcategory_index += 1
                
    for word in content_count:
        if content_count[word]<3:
            continue
        content_dict[word] = content_index
        content_index += 1
        
    return news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict


def get_doc_input(news,
                  news_index,
                  category_dict,
                  subcategory_dict,
                  entity_dict,
                  word_dict,
                  content_dict,
                  news_pop_cnt,
                  pop_max,
                  drop_target=True,
                  drop_ratio=0.2):

    news_num = len(news)+1
    news_title = np.zeros((news_num, MAX_TITLE), dtype='int32')
    news_vert = np.zeros((news_num,), dtype='int32')
    news_subvert = np.zeros((news_num,), dtype='int32')
    news_entity = np.zeros((news_num, MAX_ENTITY), dtype='int32')
    news_content = np.zeros((news_num, MAX_CONTENT), dtype='int32')
    news_pop = np.zeros((news_num, 1), dtype='int32')

    tar_title_inp = np.zeros((news_num, MAX_TITLE), dtype='int32')
    tar_title_real = np.zeros((news_num, MAX_TITLE), dtype='int32')
    
    for key in news:
        vert, subvert, title, entity, content = news[key]
        doc_index = news_index[key]

        if key in news_pop_cnt:
            news_pop[doc_index] = news_pop_cnt[key]
        else:
            news_pop[doc_index] = pop_max
        news_vert[doc_index] = category_dict[vert]
        news_subvert[doc_index] = subcategory_dict[subvert]

        for word_id in range(min(MAX_TITLE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id]]

        ll = min(MAX_TITLE,len(title))
        
        tar_title_inp[doc_index, :ll-1] = news_title[doc_index,:ll-1]
        tar_title_real[doc_index, :ll-1] = news_title[doc_index,1:ll]

        if drop_target:
            mask_target_index = np.random.choice(
                ll-1, int((ll-1)*drop_ratio), replace=False)
            for w in mask_target_index:
                tar_title_inp[doc_index, w] = 0

        for content_id in range(min(MAX_ENTITY,len(content))):
            if not content[content_id] in content_dict:
                continue
            news_content[doc_index,content_id]=content_dict[content[content_id]]                
                
        for entity_id in range(min(MAX_ENTITY, len(entity))):
            news_entity[doc_index, entity_id] = entity_dict[entity[entity_id]]

    return news_title, news_vert, news_subvert, news_entity, news_content, news_pop, tar_title_inp, tar_title_real


def read_train_clickhistory(news_index, file_path, filter_num):
    
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
            
        if len(true_click)<filter_num:
            continue
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def read_test_clickhistory(news_index,file_path,filter_num):
    
    lines = []
    with open(file_path) as f:  
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)

        if len(true_click)<filter_num:
            continue
            
            
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions


def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_CLICK),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_CLICK:
            click = click[-MAX_CLICK:]
        else:
            click=[0]*(MAX_CLICK-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,NPRATIO)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)

    sess_all = np.zeros((len(sess_pos),1+NPRATIO),dtype='int32')
    label = np.zeros((len(sess_pos),1+NPRATIO))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label


def get_test_input(news_index,session):
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word