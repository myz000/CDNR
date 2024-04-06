from tensorflow.keras.utils import Sequence
import numpy as np


class get_hir_train_generator(Sequence):
    def __init__(self, news_scoring, clicked_news, user_id, news_id, label, batch_size,
                 vae_news_scoring, vae_news_num):
        self.news_emb = news_scoring
        self.vae_news_scoring = vae_news_scoring
        self.vae_news_num = vae_news_num
        
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        news_emb = self.news_emb[docids]

        return news_emb

    def __get_vae_news(self, docids):
        news_emb = self.vae_news_scoring[docids]
        return news_emb

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]

        doc_ids = self.doc_id[start:ed]
        title = self.__get_news(doc_ids)

        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        
        user_title = self.__get_news(clicked_ids)
        
        clicked_ids_list = np.array(list(set(clicked_ids.flatten())))
        clicked_ids_list = clicked_ids_list[clicked_ids_list != 0]
        vae_clicked_ids_index = np.random.choice(len(clicked_ids_list),self.batch_size*self.vae_news_num)
        vae_clicked_ids = clicked_ids_list[vae_clicked_ids_index]
        vae_title = self.__get_vae_news(vae_clicked_ids)
        vae_title = vae_title.reshape((self.batch_size, self.vae_news_num,-1))
        
        return ([title, user_title,vae_title], [label])


class get_test_generator(Sequence):
    def __init__(self, news_scoring, docids, userids, clicked_news, batch_size):
        self.docids = docids
        self.userids = userids

        self.news_scoring = news_scoring
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.docids.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        news_scoring = self.news_scoring[docids]

        return news_scoring

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        docids = self.docids[start:ed]

        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        title = self.__get_news(docids)
        user_title = self.__get_news(clicked_ids)

        return [title, user_title,]
    
    
class get_user_generator(Sequence):
    def __init__(self, news_scoring, userids, clicked_news,batch_size):
        self.userids = userids
        self.news_scoring = news_scoring
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self,docids):
        news_scoring = self.news_scoring[docids]
        
        return news_scoring
              
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        
        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        user_title = self.__get_news(clicked_ids)

        return user_title