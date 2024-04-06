import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class SelfAttention(layers.Layer):

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
        
        Q_seq = tf.matmul(Q_seq, self.WQ)  #K.dot(Q_seq, self.WQ)
        Q_seq = tf.reshape(
            Q_seq, [-1, get_shape(Q_seq)[1], self.multiheads, self.head_dim]
        )
        Q_seq = tf.transpose(Q_seq,[0,2,1,3])

        K_seq = tf.matmul(K_seq, self.WK) #K.dot(K_seq, self.WK)
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
    
class Pooler(tf.keras.Model):
    def __init__(self, d_model, name):
        super().__init__(name = name)
        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')

    def call(self, x, mask):
        projected = self.attention_layer(x)  
        logits = tf.squeeze(self.attention_v(projected), 2)  
        logits += (1-mask) * -(1e9)
        scores = tf.expand_dims(tf.nn.softmax(logits), 1)  
        x = tf.squeeze(tf.matmul(scores, x), 1)   

        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])

    def call(self, x):
        x = self.seq(x)
        return x
    
class PriorNetwork(tf.keras.Model):
    def __init__(self, dff, d_latent):
        super().__init__(name = 'prior_net')
       
        self.hidden_layer = tf.keras.layers.Dense(dff // 2, activation = 'relu',
            name = 'hidden_layer')
        self.hidden_layer_mu = tf.keras.layers.Dense(dff // 4, activation = 'relu',
            name = 'hidden_layer_mu')
        self.hidden_layer_logvar = tf.keras.layers.Dense(dff // 4, activation = 'relu',
            name = 'hidden_layer_logvar')

        self.output_layer_mu = tf.keras.layers.Dense(d_latent, activation = 'tanh',
            name = 'output_layer_mu')
        self.output_layer_logvar = tf.keras.layers.Dense(d_latent, activation = 'tanh',
            name = 'output_layer_logvar')

    def call(self, inp):
        h = self.hidden_layer(inp)

        h_mu = self.hidden_layer_mu(h)
        mu = self.output_layer_mu(h_mu)

        h_logvar = self.hidden_layer_logvar(h)
        logvar = self.output_layer_logvar(h_logvar)

        z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(logvar))

        return z, mu, logvar
    
class RecognitionNetwork(tf.keras.Model):
    def __init__(self, dff, d_latent):
        super().__init__(name = 'recog_net')
        self.hidden_layer = tf.keras.layers.Dense(dff, activation = 'relu',
            name = 'hidden_layer')
        self.hidden_layer_mu = tf.keras.layers.Dense(dff // 2, activation =  'relu',
            name = 'hidden_layer_mu')
        self.hidden_layer_logvar = tf.keras.layers.Dense(dff // 2, activation =  'relu',
            name = 'hidden_layer_logvar')

        self.output_layer_mu = tf.keras.layers.Dense(d_latent, activation = 'tanh',
            name = 'output_layer_mu')
        self.output_layer_logvar = tf.keras.layers.Dense(d_latent, activation = 'tanh',
            name = 'output_layer_logvar')

    def call(self, cond, inp):       
        x = tf.concat([cond,inp], axis = -1)        
        h = self.hidden_layer(x)
        h_mu = self.hidden_layer_mu(h)       
        mu = self.output_layer_mu(h_mu)
        h_logvar = self.hidden_layer_logvar(h)
        logvar = self.output_layer_logvar(h_logvar)
        z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(logvar))
        return z, mu, logvar
    
class BowNetwork(tf.keras.Model):
    def __init__(self, dff, vocab_size, tar_seq_len):
        super().__init__(name = 'bow_net')
        self.hidden_layer = tf.keras.layers.Dense(dff, activation = 'relu',name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(vocab_size, name = 'output_layer')
        self.tar_seq_len = tar_seq_len

    def call(self, x):
        h = self.hidden_layer(x) 
        bow_logits = self.output_layer(h)        
        bow_logits = tf.tile(tf.expand_dims(bow_logits, 1),[1,self.tar_seq_len,1])       
        return bow_logits
    
class ComputeMasking(tf.keras.layers.Layer):
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
    
class Embedder(tf.keras.Model):
    def __init__(self, d_model, dropout_rate, word_embedding_matrix,max_position_embed, seq_len):
        super().__init__(name = 'embedder')

        self.padding_idx = 1
        self.seq_len = seq_len

        self.word_embeddings = tf.keras.layers.Embedding(word_embedding_matrix.shape[0],
                                                         word_embedding_matrix.shape[1],
                                                         weights = [word_embedding_matrix],
                                                         trainable = True,
                                                         name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')

        self.layernorm = tf.keras.layers.LayerNormalization(name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

    def call(self, x):
        seq_len = self.seq_len
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, get_shape(x))
        x = self.word_embeddings(x)  
        x += self.pos_embeddings(pos)
        x = self.layernorm(x)
        x = self.dropout(x)

        return x  
    
class Encoder_Layer(tf.keras.layers.Layer):
    def __init__(self, d_model, head_num, dff, dropout_rate, **kwargs):
        self.mha = SelfAttention(head_num,int(d_model/head_num), mask_right=False)
        
        self.ffn = FeedForward(d_model,dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout')
        
        super(Encoder_Layer, self).__init__(**kwargs)

    def call(self, x, mask):
        selfatt_out = self.mha([x,x,x],mask)
        selfatt_out = self.dropout(selfatt_out)        
        selfatt_out = self.layernorm1(x + selfatt_out)
        ffn_output = self.ffn(selfatt_out)
        ffn_output = self.layernorm2(selfatt_out+ffn_output)
        return ffn_output 

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, dff, head_num, dropout_rate,**kwargs):
        self.num_layers = num_layers
        self.enc_layers = [
             Encoder_Layer(d_model, head_num, dff, dropout_rate)for _ in range(num_layers)
        ]
        
        super(Encoder, self).__init__(**kwargs)

    def call(self, x, mask):     
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)                
        return x  

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_latent, head_num, dff,dropout_rate, seq_len, **kwargs):
        self.seq_len = seq_len
        
        self.mha = SelfAttention(head_num,int(d_model/head_num), mask_right=True)
        
        self.mask_layer = ComputeMasking()
        
        self.ffn = FeedForward(d_model,dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        super(Decoder_Layer, self).__init__(**kwargs)

    def call(self, x, z, cond, x_mask):
        seq_len = self.seq_len
        selfatt_out = self.mha([x,x,x],x_mask) 
        selfatt_out = self.dropout1(selfatt_out) 
        out = self.layernorm1(x + selfatt_out)      
        con_out = tf.tile(tf.expand_dims(cond,1),[1,seq_len,1]) 
        out2 = self.layernorm2(out + con_out)           
        z = tf.tile(tf.expand_dims(z,1),[1,seq_len,1]) 
        out3= tf.concat([out2,z],axis=-1)     
        ffn_output = self.ffn(out3)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.layernorm3(out2+ffn_output)               
        return ffn_output  
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_latent, num_heads, dff, dropout_rate,seq_len):
        super().__init__(name = 'decoder')
        self.num_layers = num_layers
        self.dec_layers = [
            Decoder_Layer(d_model, d_latent, num_heads, dff, dropout_rate,seq_len)
            for i in range(num_layers)
        ]

    def call(self, x, z, cond, x_mask):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x,z,cond,x_mask)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    
class CVAETransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_latent, head_num, dff, dropout_rate, 
                 max_position_embed, vocab_size, pop_range, word_embedding_matrix, seq_len):
        super().__init__(name = 'cvae_transformer')
        
        
        self.pop_embeddings = tf.keras.layers.Embedding(pop_range, d_model, name = 'pop_embed')
        
        self.embedder = Embedder(d_model, dropout_rate, word_embedding_matrix, max_position_embed,seq_len)
        
        self.encoder = Encoder(num_layers, d_model, d_model, head_num, dropout_rate)
        self.pooler = Pooler(d_model,'pool')
        
        self.prior_net = PriorNetwork(dff, d_latent*2)
        self.recog_net = RecognitionNetwork(dff, d_latent*2)
        self.bow_net = BowNetwork(dff, vocab_size, seq_len)
        
        self.mask_layer = ComputeMasking()

        self.decoder = Decoder(num_layers, d_model, d_latent, head_num, dff, dropout_rate,seq_len)

        self.final_layer = tf.keras.layers.Dense(vocab_size, name = 'final_layer')
        
        self.d_latent = d_latent

    def call(self, inp, cond, tar):
        inp_mask = self.mask_layer(inp)
        tar_mask = self.mask_layer(tar)
        inp_embed = self.embedder(inp)
        tar_embed = self.embedder(tar)       
        cond_embed = self.pop_embeddings(cond)
        cond_embed = tf.squeeze(cond_embed,1)
        
        
        enc_inp_output = self.encoder(inp_embed,inp_mask)
        enc_inp_output_pooled = self.pooler(enc_inp_output, inp_mask)
               
        _, mu_p, logvar_p = self.prior_net(cond_embed)  
        z, mu_r, logvar_r = self.recog_net(cond_embed, enc_inp_output_pooled) 
        
        dec_output = self.decoder(tar_embed, z[:,:self.d_latent], cond_embed,tar_mask)
        dec_logits = self.final_layer(dec_output)
        bow_inp = z[:,:self.d_latent]
        bow_logits = self.bow_net(bow_inp) 
        
        return dec_logits, mu_r, logvar_r, mu_p, logvar_p, bow_logits, z,