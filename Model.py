from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from CVAETransformer import *
from Hypers import *

def AttentivePooling(dim1, dim2,drop_ratio):
    vecs_input = Input(shape=(dim1, dim2), dtype='float32')
    mask = Input(shape=(dim1, ), dtype='float32')
    user_vecs = Dropout(drop_ratio)(vecs_input)
    user_att = Dense(200, activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att += (1-mask)* -(1e9)
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1, 1))([user_vecs, user_att])
    model = Model([vecs_input,mask], user_vec)
    return model


def get_transformer_encoder(Transformer):
    mask_layer = Transformer.mask_layer
    embedder = Transformer.embedder
    pop_embeddings = Transformer.pop_embeddings
    encoder = Transformer.encoder
    pooler = Transformer.pooler
    recog_net = Transformer.recog_net

    sentence_input = Input(shape=(MAX_TITLE+1,), dtype='int32')
    inp = keras.layers.Lambda(lambda x: x[:, :MAX_TITLE])(sentence_input)
    cond = keras.layers.Lambda(lambda x: x[:, MAX_TITLE:])(sentence_input)

    inp_mask = mask_layer(inp)
    inp_embed = embedder(inp)

    cond_embed = pop_embeddings(cond)
    cond_embed = tf.squeeze(cond_embed, 1)
 
    enc_inp_output = encoder(inp_embed, inp_mask)
    enc_inp_output_pooled = pooler(enc_inp_output, inp_mask)

    z, mu_r, logvar_r = recog_net(cond_embed, enc_inp_output_pooled)  # (z|c,x)


    news_encoder = keras.Model(sentence_input, mu_r)
    return news_encoder

class SelfAttention(Layer):
    def __init__(self, multiheads=3, head_dim=300, seed=0, mask_right=False, **kwargs):
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
            regularizer = keras.regularizers.l2(0.0001)
            
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
            regularizer = keras.regularizers.l2(0.0001)
            
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
            regularizer = keras.regularizers.l2(0.0001)
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, mask, mode="add"):
        if mask == None:
            return inputs
        else:

            for _ in range(len(inputs.shape) - 2):
                mask = tf.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs,masks):
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        
        Q_seq = tf.matmul(Q_seq, self.WQ)  
        Q_seq = tf.reshape(
            Q_seq, [-1, get_shape(Q_seq)[1], self.multiheads, self.head_dim]
        )
        Q_seq = tf.transpose(Q_seq,[0,2,1,3])
       
        K_seq = tf.matmul(K_seq, self.WK) 
        K_seq = tf.reshape(
            K_seq, [-1, get_shape(K_seq)[1], self.multiheads, self.head_dim]
        )
        K_seq = tf.transpose(K_seq, [0, 2, 1, 3])

        V_seq = tf.matmul(V_seq, self.WV)
        V_seq = tf.reshape(
            V_seq, [-1, get_shape(V_seq)[1], self.multiheads, self.head_dim]
        )
        V_seq = tf.transpose(V_seq, [0, 2, 1, 3])
        A = tf.matmul(Q_seq, K_seq, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        A = tf.transpose(
            A, [0, 3, 2, 1]
        ) 

        A = self.Mask(A, masks, "add")
        A = tf.transpose(A, [0, 3, 2, 1])
        if self.mask_right:
            ones = tf.ones_like(A[:1, :1])
            lower_triangular = tf.linalg.band_part(ones, -1, 0)
            mas = (ones - lower_triangular) * 1e12
            A = A - mas
        A = tf.nn.softmax(A)

        O_seq = tf.matmul(A, V_seq)
        O_seq = tf.transpose(O_seq, [0, 2, 1, 3])

        O_seq = tf.reshape(O_seq, [-1, get_shape(O_seq)[1], self.output_dim])
        O_seq = self.Mask(O_seq, masks, "mul")
        return O_seq

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config

    def compute_output_shape(self, input_shape):        
        return (input_shape[0][0], input_shape[0][-2], self.output_dim)
    
    
class ComputeMasking(Layer):
    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):   
        config = super().get_config().copy()
        return config


def get_user_encoder(news_encoder,d_model,drop_ratio):
    clicked_title_input =  Input(shape=(MAX_CLICK,MAX_TITLE+1,), dtype='int32')
    mask = ComputeMasking()(tf.reduce_sum(clicked_title_input,-1))
    clicked_news_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    clicked_news_vecs = Dropout(drop_ratio)(clicked_news_vecs) 
    user_vec_1 = SelfAttention(4,int(d_model/4))([clicked_news_vecs[:,:,:d_model]]*3,mask)
    user_vec_1 = Dropout(drop_ratio)(user_vec_1)
    user_vec_1 = AttentivePooling(MAX_CLICK,d_model,drop_ratio)([user_vec_1,mask])

    user_vec_2 = SelfAttention(4,int(d_model/4))([clicked_news_vecs[:,:,d_model:]]*3,mask)
    user_vec_2 = Dropout(drop_ratio)(user_vec_2)
    user_vec_2 = AttentivePooling(MAX_CLICK,d_model,drop_ratio)([user_vec_2,mask])
    user_vec = Concatenate(axis=-1)([user_vec_1,user_vec_2])
    
    model = Model(clicked_title_input,user_vec)
    return model 


def init_Transformer(word_embedding_matrix,num_layers,d_model,d_latent,num_heads,dropout_rate,max_position,pop_box_nums):
    dff = d_model * 4
    max_position_embed = max_position
    vocab_size = word_embedding_matrix.shape[0]
    pop_range = pop_box_nums
    seq_len = MAX_TITLE
    Transformer = CVAETransformer(num_layers, d_model, d_latent, num_heads, dff, dropout_rate, 
                 max_position_embed, vocab_size, pop_range, word_embedding_matrix, seq_len)
    return Transformer


class CVAE_Loss(Layer):
    def __init__(self, **kwargs):
        super(CVAE_Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        
        real, dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, lambda_reco, lambda_kl, lambda_bow = inputs
        
        mask = real != 0        
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')        
        dec_loss = scce(real, dec_logits)
        mask = tf.cast(mask, dtype = dec_loss.dtype)
        dec_loss *= mask
        dec_loss = tf.reduce_sum(dec_loss, -1) / tf.reduce_sum(mask, -1)
        
        kl_div = 0.5 * tf.reduce_sum(logvar_p - logvar_r - 1
                                     + tf.exp(logvar_r - logvar_p)
                                     + (mu_p - mu_r) ** 2 / tf.exp(logvar_p), axis = -1)

        
        kl_div = tf.math.maximum(5.0,kl_div)
        
        bow_loss = scce(real, bow_logits) * mask
        bow_loss = tf.reduce_sum(bow_loss, -1) / tf.reduce_sum(mask, -1)

        return dec_loss, kl_div, bow_loss
    

def create_model(word_embedding_matrix, 
                 num_layers,
                 d_model, 
                 d_latent,
                 num_heads,
                 news_encoder_train, 
                 dropout_rate,
                 lambda_reco, 
                 lambda_kl, 
                 lambda_bow,
                 max_position,
                 pop_box_nums):
    
    Transformer = init_Transformer(word_embedding_matrix,num_layers,d_model,d_latent,num_heads,dropout_rate,max_position,pop_box_nums)

    news_encoder = get_transformer_encoder(Transformer)   
    news_encoder.compute_output_shape = lambda x: (x[0], d_latent*2)
    news_encoder.trainable = news_encoder_train
    user_encoder = get_user_encoder(news_encoder,d_latent,dropout_rate)

    clicked_title_input = Input(shape=(MAX_CLICK, MAX_TITLE+1,), dtype='int32')
    title_inputs = Input(shape=(1+NPRATIO, MAX_TITLE+1,), dtype='int32')
    vae_title_inputs = Input(
        shape=(MAX_VAE_TITLE, MAX_TITLE*3+1,), dtype='int32')

    user_evecs = user_encoder(clicked_title_input)  
    user_evecs = tf.tile(tf.expand_dims(user_evecs, 1), [1, 1+NPRATIO, 1])
    news_evecs = TimeDistributed(news_encoder)(title_inputs)  
    scores = tf.reduce_sum(tf.multiply(user_evecs, news_evecs), -1)

    logits = keras.layers.Activation(
        keras.activations.softmax, name='recommend')(scores)

    vae_title = tf.reshape(vae_title_inputs,[-1, MAX_TITLE*3+1])
    vae_inp_arr = keras.layers.Lambda(lambda x: x[:,:MAX_TITLE])(vae_title)
    tar_inp_arr = keras.layers.Lambda(
        lambda x: x[:,MAX_TITLE:MAX_TITLE*2])(vae_title)
    con_arr = keras.layers.Lambda(
        lambda x: x[:,MAX_TITLE*2:MAX_TITLE*2+1])(vae_title)
    tar_real_arr = keras.layers.Lambda(lambda x: x[:,MAX_TITLE*2+1:])(vae_title)
    dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, z, = Transformer(
        vae_inp_arr, con_arr, tar_inp_arr)
    
    
    dec_loss, kl_div, bow_loss = CVAE_Loss()(
        [tar_real_arr, dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, lambda_reco, lambda_kl, lambda_bow])
    model = Model([title_inputs, clicked_title_input,vae_title_inputs,],
                  logits)  
    
    dec_loss = tf.reduce_mean(dec_loss)
    kl_div = tf.reduce_mean(kl_div)
    bow_loss = tf.reduce_mean(bow_loss)

    
    model.add_loss(lambda_reco*dec_loss, inputs=True)
    model.add_metric(lambda_reco*dec_loss, aggregation="mean", name="dec_loss")

    model.add_loss(lambda_kl * kl_div, inputs=True)
    model.add_metric(lambda_kl * kl_div, aggregation="mean", name="kl_div")

    model.add_loss(lambda_bow * bow_loss, inputs=True)
    model.add_metric(lambda_bow * bow_loss, aggregation="mean", name="bow_loss")
    
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.CategoricalCrossentropy(),
                           tf.keras.metrics.categorical_accuracy])
    model1 = Model([title_inputs, clicked_title_input,vae_title_inputs,],
                   [scores,logits,dec_loss, kl_div, bow_loss])  

    return model, news_encoder, model1, user_encoder,Transformer