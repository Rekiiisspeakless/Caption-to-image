#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

import scipy
from scipy.io import loadmat
import re

import string
import imageio
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import random
import time
import nltk

import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

warnings.filterwarnings('ignore')

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.5


# In[2]:


dictionary_path = './dictionary'
vocab = np.load(dictionary_path+'/vocab.npy')
print('there are {} vocabularies in total'.format(len(vocab)))

word2Id_dict = dict(np.load(dictionary_path+'/word2Id.npy'))
id2word_dict =  dict(np.load(dictionary_path+'/id2Word.npy'))
print('Word to id mapping, for example: %s -> %s'%('flower', word2Id_dict['flower']))
print('Id to word mapping, for example: %s -> %s'%('2428', id2word_dict['2428']))
print('Tokens: <PAD>: %s; <RARE>: %s'%(word2Id_dict['<PAD>'], word2Id_dict['<RARE>']))


# In[3]:


def sent2IdList(line, MAX_SEQ_LENGTH=20):
    MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
    padding = 0
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    tokens = []
    tokens.extend(nltk.tokenize.word_tokenize(prep_line.lower()))
    l = len(tokens)
    padding = MAX_SEQ_LIMIT - l
    for i in range(padding):
        tokens.append('<PAD>')
    line = [word2Id_dict[tokens[k]] if tokens[k] in word2Id_dict else word2Id_dict['<RARE>'] for k in range(len(tokens))]
    
    return line

#nltk.download('punkt')
text = "the flower shown has yellow anther red pistil and bright red petals."
print(text)
print(sent2IdList(text))


# In[4]:


data_path = './dataset'
df = pd.read_pickle(data_path+'/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data'%(n_images_train))


# In[5]:


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_DEPTH = 3
def training_data_generator(caption, image_path):
    # load in the image according to image path
    imagefile = tf.read_file(image_path)
    image = tf.image.decode_image(imagefile, channels=3)
    float_img = tf.image.convert_image_dtype(image, tf.float32)
    float_img.set_shape([None, None, 3])
    image = tf.image.resize_images(float_img, size = [IMAGE_HEIGHT, IMAGE_WIDTH])
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    
    return image, caption

def data_iterator(filenames, batch_size, data_generator):
    # Load the training data into two NumPy arrays
    df = pd.read_pickle(filenames)
    captions = df['Captions'].values
    caption = []
    for i in range(len(captions)):
        caption.append(random.choice(captions[i])) 
    caption = np.asarray(caption)
    #print(df['ImagePath'].values.shape)
    image_path = 'dataset/'+df['ImagePath'].values
    
    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert caption.shape[0] == df['ImagePath'].values.shape[0]
    
    dataset = tf.data.Dataset.from_tensor_slices((caption, image_path))
    dataset = dataset.map(data_generator)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes
    
    return iterator, output_types, output_shapes


# In[6]:


tf.reset_default_graph()
BATCH_SIZE = 64
iterator_train, types, shapes = data_iterator(data_path+'/text2ImgData.pkl', BATCH_SIZE, training_data_generator)
iter_initializer = iterator_train.initializer
next_element = iterator_train.get_next()

with tf.Session(config = config) as sess:
    sess.run(iterator_train.initializer)
    next_element = iterator_train.get_next()
    image, text = sess.run(next_element)


# In[84]:


class TextEncoder:
    """
    Encode text (a caption) into hidden representation
    input: text (a list of id)
    output: hidden representation of input text in dimention of TEXT_DIM
    """
    def __init__(self, text, hparas, training_phase=True, reuse=False, return_embed=False):
        self.text = text
        self.hparas = hparas
        self.train = training_phase
        self.reuse = reuse
        self._build_model()
    def _build_model(self):
        with tf.variable_scope('rnnftxt', reuse=self.reuse):
            # Word embedding
            word_embed_matrix = tf.get_variable('rnn/wordembed', 
                                                shape=(self.hparas['VOCAB_SIZE'], self.hparas['EMBED_DIM']),
                                                initializer=tf.random_normal_initializer(stddev=0.02),
                                                dtype=tf.float32)
            embedded_word_ids = tf.nn.embedding_lookup(word_embed_matrix, self.text)
#             network = EmbeddingInputlayer(
#                                  inputs = self.text,
#                                  vocabulary_size = self.hparas['VOCAB_SIZE'],
#                                  embedding_size = self.hparas['EMBED_DIM'],
#                                  E_init = tf.random_normal_initializer(stddev=0.02),
#                                  name = 'rnn/wordembed')
            # RNN encoder
            LSTMCell = tf.nn.rnn_cell.LSTMCell(self.hparas['TEXT_DIM'], reuse=self.reuse)
            LSTMCell = tf.nn.rnn_cell.DropoutWrapper(LSTMCell, input_keep_prob = self.hparas['KEEP_PROB'], output_keep_prob = self.hparas['KEEP_PROB'])
            initial_state = LSTMCell.zero_state(self.hparas['BATCH_SIZE'], dtype=tf.float32)
            rnn_net = tf.nn.dynamic_rnn(cell=LSTMCell, 
                                        inputs=embedded_word_ids, 
                                        initial_state=initial_state, 
                                        dtype=np.float32, time_major=False,
                                        scope='rnn/dynamic')
            
            
#             network = DynamicRNNLayer(network,
#                      cell_fn = LSTMCell,
#                      cell_init_args = {'state_is_tuple' : True, 'reuse': self.reuse},  # for TF1.1, TF1.2 dont need to set reuse
#                      n_hidden = self.hparas['RNN_HIDDEN_SIZE'],
#                      dropout = (self.hparas['KEEP_PROB'] if self.train else None),
#                      initializer = tf.random_normal_initializer(stddev=0.02),
#                      sequence_length = tl.layers.retrieve_seq_length_op2(self.text),
#                      return_last = True,
#                      name = 'rnn/dynamic')
            self.rnn_net = rnn_net
            self.outputs = rnn_net[0][:, -1, :]


# In[29]:


class ImageEncoder:
    def __init__(self, image, hparas, training_phase=True, reuse=False, return_embed=False):
        self.image = image
        self.hparas = hparas
        self.train = training_phase
        self.reuse = reuse
        self._build_model()
    def _build_model(self):
        kernal_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        df_dim = 64
        with tf.variable_scope('cnnftxt', reuse=self.reuse):
            net0 = tf.layers.conv2d(self.image, df_dim, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'cnn/conv2d1')
            net1 = tf.layers.conv2d(net0, df_dim*2, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'cnn/conv2d2')
            net1 = tf.layers.batch_normalization(net1, training = self.train, gamma_initializer = gamma_init, name='cnn/batch_norm1')
            net1 = tf.nn.leaky_relu(net1, 0.2)
            net2 = tf.layers.conv2d(net1, df_dim*4, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'cnn/conv2d3')
            net2 = tf.layers.batch_normalization(net2, training = self.train, gamma_initializer = gamma_init, name='cnn/batch_norm2')
            net2 = tf.nn.leaky_relu(net2, 0.2)
            net3 = tf.layers.conv2d(net2, df_dim*8, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'cnn/conv2d4')
            net3 = tf.layers.batch_normalization(net3, training = self.train, gamma_initializer = gamma_init, name='cnn/batch_norm3')
            
            net4 = tf.contrib.layers.flatten(net3)
            net4 = tf.layers.dense(net4, self.hparas['TEXT_DIM'], name='cnn/dense', reuse=self.reuse)
            
            self.outputs = net4
            


# In[30]:


class Generator:
    def __init__(self, noise_z, text, training_phase, hparas, reuse):
        self.z = noise_z
        self.text = text
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = 128
        self.reuse = reuse
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('generator', reuse=self.reuse):
            text_flatten = tf.contrib.layers.flatten(self.text)
            text_input = tf.layers.dense(text_flatten, self.hparas['TEXT_DIM'], name='generator0/text_input', reuse=self.reuse)
            text_input = tf.nn.leaky_relu(text_input, 0.2)
            z_text_concat = tf.concat([self.z, text_input], axis=1, name='generator0/z_text_concat')
            
            gamma_init = tf.random_normal_initializer(1., 0.02)
            kernal_init = tf.random_normal_initializer(stddev=0.02)
            gf_dim = 128
            
            g_net0 = tf.layers.dense(z_text_concat, gf_dim*8*4*4, name='generator0/g_net', reuse=self.reuse)
            g_net0 = tf.layers.batch_normalization(g_net0, training = self.train, gamma_initializer = gamma_init, name='generator0/batch_norm')
            g_net0 = tf.reshape(g_net0, [-1, 4, 4, gf_dim*8], name='generator0/g_net_reshape')
            
            ######1
            g_net = tf.layers.conv2d(g_net0, gf_dim*2, (1, 1), (1, 1), padding = 'VALID', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator1/res/conv2d1')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator1/batch_norm1')
            g_net = tf.nn.relu(g_net)
            g_net = tf.layers.conv2d(g_net, gf_dim*2, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator1/res/conv2d2')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator1/batch_norm2')
            g_net = tf.nn.relu(g_net)
            g_net = tf.layers.conv2d(g_net, gf_dim*8, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator1/res/conv2d3')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator1/batch_norm3')
            g_net1 = tf.add(g_net, g_net0)
            g_net1 = tf.nn.relu(g_net1)
            
            ######2
            g_net2 = tf.image.resize_images(g_net1, [8, 8], method = 1, align_corners = False)
            g_net2 = tf.layers.conv2d(g_net2, gf_dim*4, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator2/conv2d1')
            g_net2 = tf.layers.batch_normalization(g_net2, training = self.train, gamma_initializer = gamma_init, name='generator2/batch_norm1')
            g_net2 = tf.nn.relu(g_net2)
            
            ######3
            g_net = tf.layers.conv2d(g_net2, gf_dim, (1, 1), (1, 1), padding = 'VALID', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator3/res/conv2d1')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator3/batch_norm1')
            g_net = tf.nn.relu(g_net)
            g_net = tf.layers.conv2d(g_net, gf_dim, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator3/res/conv2d2')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator3/batch_norm2')
            g_net = tf.nn.relu(g_net)
            g_net = tf.layers.conv2d(g_net, gf_dim*4, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator3/res/conv2d3')
            g_net = tf.layers.batch_normalization(g_net, training = self.train, gamma_initializer = gamma_init, name='generator3/batch_norm3')
            g_net3 = tf.add(g_net, g_net2)
            g_net3 = tf.nn.relu(g_net3)
            
            #####4
            g_net4 = tf.image.resize_images(g_net3, [16, 16], method = 1, align_corners = False)
            g_net4 = tf.layers.conv2d(g_net4, gf_dim*2, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator4/conv2d1')
            g_net4 = tf.layers.batch_normalization(g_net4, training = self.train, gamma_initializer = gamma_init, name='generator4/batch_norm1')
            g_net4 = tf.nn.relu(g_net4)
            
            #####5
            g_net5 = tf.image.resize_images(g_net4, [32, 32], method = 1, align_corners = False)
            g_net5 = tf.layers.conv2d(g_net5, gf_dim, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generator5/conv2d1')
            g_net5 = tf.layers.batch_normalization(g_net5, training = self.train, gamma_initializer = gamma_init, name='generator5/batch_norm1')
            g_net5 = tf.nn.relu(g_net5)
            
            #####output
            g_neto = tf.image.resize_images(g_net5, [64, 64], method = 1, align_corners = False)
            g_neto = tf.layers.conv2d(g_neto, 3, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'generatoro/conv2d1')
            
            
            self.generator_net = g_neto
            self.outputs = tf.nn.tanh(g_neto)
            self.logits = g_neto


# In[31]:


# resnet structure
class Discriminator:
    def __init__(self, image, text, training_phase, hparas, reuse):
        self.image = image
        self.text = text
        self.train = training_phase
        self.hparas = hparas
        self.df_dim = 128 # 196 for MSCOCO
        self.reuse = reuse
        self._build_model()
    
    def _build_model(self):        
        with tf.variable_scope('discriminator', reuse=self.reuse):
            kernal_init = tf.random_normal_initializer(stddev=0.02)
            gamma_init=tf.random_normal_initializer(1., 0.02)
            df_dim = 64
            
            text_flatten = tf.contrib.layers.flatten(self.text)
            text_input = tf.layers.dense(text_flatten, self.hparas['TEXT_DIM'], name='discrim/text_input', reuse=self.reuse)
            text_input = tf.nn.leaky_relu(text_input, 0.2)
            text_input = tf.expand_dims(text_input, axis=1)
            text_input = tf.expand_dims(text_input, axis=1)
            text_input = tf.tile(text_input, multiples=[1,4,4,1])
            
            
            
            #image_flatten = tf.contrib.layers.flatten(self.image)
            #image_input = tf.layers.dense(image_flatten, self.hparas['TEXT_DIM'], name='discrim/image_input', reuse=self.reuse)
            net0 = tf.layers.conv2d(self.image, df_dim, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim0/image/conv2d1')
            
            net1 = tf.layers.conv2d(net0, df_dim*2, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim1/image/conv2d1')
            net1 = tf.layers.batch_normalization(net1, training = self.train, gamma_initializer = gamma_init, name='discrim1/image/batch_norm1')
            net1 = tf.nn.leaky_relu(net1, 0.2)
            net2 = tf.layers.conv2d(net1, df_dim*4, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim2/image/conv2d1')
            net2 = tf.layers.batch_normalization(net2, training = self.train, gamma_initializer = gamma_init, name='discrim2/image/batch_norm1')
            net2 = tf.nn.leaky_relu(net2, 0.2)
            net3 = tf.layers.conv2d(net2, df_dim*8, (4, 4), (2, 2), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim3/image/conv2d1')
            net3 = tf.layers.batch_normalization(net3, training = self.train, gamma_initializer = gamma_init, name='discrim3/image/batch_norm1')
            
            net = tf.layers.conv2d(net3, df_dim*2, (1, 1), (1, 1), padding = 'VALID', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim4/image/conv2d1')
            net = tf.layers.batch_normalization(net, training = self.train, gamma_initializer = gamma_init, name='discrim4/image/batch_norm1')
            net = tf.nn.leaky_relu(net, 0.2)
            net = tf.layers.conv2d(net, df_dim*2, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim4/image/conv2d2')
            net = tf.layers.batch_normalization(net, training = self.train, gamma_initializer = gamma_init, name='discrim4/image/batch_norm2')
            net = tf.nn.leaky_relu(net, 0.2)
            net = tf.layers.conv2d(net, df_dim*8, (3, 3), (1, 1), padding = 'SAME', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim4/image/conv2d3')
            net = tf.layers.batch_normalization(net, training = self.train, gamma_initializer = gamma_init, name='discrim4/image/batch_norm3')
            
            net4 = tf.add(net, net3)
            net4 = tf.nn.leaky_relu(net4, 0.2)
            
            img_text_concate = tf.concat([text_input, net4], axis=3, name='discrim/concate')
            d_net = tf.layers.dense(img_text_concate, 1, name='discrim/d_net', reuse=self.reuse)
            d_net = tf.layers.conv2d(d_net, df_dim*8, (1, 1), (1, 1), padding = 'VALID', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrim/conv2d1')
            d_net = tf.layers.batch_normalization(d_net, training = self.train, gamma_initializer = gamma_init, name='discrim/batch_norm1')
            d_net = tf.nn.leaky_relu(d_net, 0.2)
            
            d_neto = tf.layers.conv2d(d_net, 1, (4, 4), (4, 4), padding = 'VALID', kernel_initializer = kernal_init,
                                bias_initializer = None, name = 'discrimo/conv2d1')
            
            self.logits = d_neto
            net_output = tf.nn.sigmoid(d_neto)
            self.discriminator_net = net_output
            self.outputs = net_output


# In[59]:


def get_hparas():
    hparas = {
        'MAX_SEQ_LENGTH' : 20,
        'EMBED_DIM' : 200, # word embedding dimension
        'VOCAB_SIZE' : len(vocab),
        'TEXT_DIM' : 200, # text embedding dimension
        'RNN_HIDDEN_SIZE' : 50,
        'KEEP_PROB' : 0.7,
        'Z_DIM' : 16, # random noise z dimension
        'IMAGE_SIZE' : [64, 64, 3], # render image size
        'BATCH_SIZE' : 64,
        'LR' : 0.0002,
        'LR_DECAY': 0.5,
        'DECAY_EVERY_EPOCH':100,
        'BETA' : 0.5, # AdamOptimizer parameter
        'N_EPOCH' : 100,
        'N_SAMPLE' : num_training_sample
    }
    return hparas


# In[101]:


class GAN:
    def __init__(self, hparas, training_phase, dataset_path, ckpt_path, inference_path, recover=None):
        self.hparas = hparas
        self.train = training_phase
        self.dataset_path = dataset_path # dataPath+'/text2ImgData.pkl'
        self.ckpt_path = ckpt_path
        self.sample_path = './samples'
        self.inference_path = './inference'
        
        self._get_session() # get session
        self._get_train_data_iter() # initialize and get data iterator
        self._input_layer() # define input placeholder
        self._get_inference() # build generator and discriminator
        self._get_loss() # define gan loss
        self._get_var_with_name() # get variables for each part of model
        self._optimize() # define optimizer
        self._init_vars()
        self._get_saver()
        
        if recover is not None:
            self._load_checkpoint(recover)
            
            
        
    def _get_train_data_iter(self):
        if self.train: # training data iteratot
            iterator_train, types, shapes = data_iterator(self.dataset_path+'/text2ImgData.pkl',
                                                          self.hparas['BATCH_SIZE'], training_data_generator)
            iter_initializer = iterator_train.initializer
            self.next_element = iterator_train.get_next()
            self.sess.run(iterator_train.initializer)
            self.iterator_train = iterator_train
        else: # testing data iterator
            iterator_test, types, shapes = data_iterator_test(self.dataset_path+'/testData.pkl', self.hparas['BATCH_SIZE'])
            iter_initializer = iterator_test.initializer
            self.next_element = iterator_test.get_next()
            self.sess.run(iterator_test.initializer)
            self.iterator_test = iterator_test
            
    def _input_layer(self):
        if self.train:
            self.real_image = tf.placeholder('float32',
                                              [self.hparas['BATCH_SIZE'], self.hparas['IMAGE_SIZE'][0],
                                               self.hparas['IMAGE_SIZE'][1], self.hparas['IMAGE_SIZE'][2]],
                                              name='real_image')
            self.caption = tf.placeholder(dtype=tf.int64, shape=[self.hparas['BATCH_SIZE'], None], name='caption')
            self.z_noise = tf.placeholder(tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']], name='z_noise')
        else:
            self.caption = tf.placeholder(dtype=tf.int64, shape=[self.hparas['BATCH_SIZE'], None], name='caption')
            self.z_noise = tf.placeholder(tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']], name='z_noise')
    
    def _get_inference(self):
        if self.train:
            # GAN training
            # encoding text
            text_encoder = TextEncoder(self.caption, hparas = self.hparas, training_phase=True, reuse=False)
            self.wrong_caption = tf.random.shuffle(self.caption)
            self.text_encoder = text_encoder
            #encoding image
            image_encoder = ImageEncoder(self.real_image, hparas = self.hparas, training_phase = True, reuse = False)
            self.image_encoder = image_encoder
            
            
            #wrong caption
            self.wrong_caption = tf.random.shuffle(self.caption)
            self.wrong_text_encoder = TextEncoder(self.wrong_caption, hparas = self.hparas, training_phase=True, reuse=True)
            #wrong image
            self.wrong_image = tf.random.shuffle(self.real_image)
            self.wrong_image_encoder = ImageEncoder(self.wrong_image, hparas = self.hparas, training_phase=True, reuse = True)
            
            
            text_encoder = TextEncoder(self.caption, hparas = self.hparas, training_phase=False, reuse=True)
            
            # generating image
            generator = Generator(self.z_noise, text_encoder.outputs, training_phase=True,
                                  hparas=self.hparas, reuse=False)
            self.generator = generator
            
            # discriminize
            # fake image real caption
            fake_discriminator = Discriminator(generator.outputs, text_encoder.outputs,
                                               training_phase=True, hparas=self.hparas, reuse=False)
            self.fake_discriminator = fake_discriminator
            # real image real caption
            real_discriminator = Discriminator(self.real_image, text_encoder.outputs, training_phase=True,
                                              hparas=self.hparas, reuse=True)
            self.real_discriminator = real_discriminator
            # real image fake caption
            real_fake_discriminator = Discriminator(self.real_image, self.wrong_text_encoder.outputs, training_phase=True,
                                              hparas=self.hparas, reuse=True)
            self.real_fake_discriminator = real_fake_discriminator
            
        else: # inference mode
            
            self.text_embed = TextEncoder(self.caption, hparas=self.hparas, training_phase=False, reuse=False)
            self.generate_image_net = Generator(self.z_noise, self.text_embed.outputs, training_phase=False,
                                                hparas=self.hparas, reuse=False)
    def _get_loss(self):
        if self.train:
#             d_loss1 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_discriminator.logits,
#                                                                               labels=tf.ones_like(self.real_discriminator.logits),
#                                                                               name='d_loss1'))
#             d_loss2 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_discriminator.logits,
#                                                                               labels=tf.zeros_like(self.fake_discriminator.logits),
#                                                                               name='d_loss2'))
#             self.d_loss = d_loss1 + d_loss2
#             self.g_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_discriminator.logits,
#                                                                                   labels=tf.ones_like(self.fake_discriminator.logits),
#                                                                                   name='g_loss'))
            rnn_t = self.text_encoder.outputs
            rnn_f = self.wrong_text_encoder.outputs
            cnn_t = self.image_encoder.outputs
            cnn_f = self.wrong_image_encoder.outputs
            alpha = 0.2
            self.rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(rnn_t, cnn_t) + cosine_similarity(rnn_t, cnn_f))) +                 tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(rnn_t, cnn_t) + cosine_similarity(rnn_f, cnn_t)))
            

            d_loss1 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_discriminator.logits,
                                                                              labels=tf.ones_like(self.real_discriminator.logits),
                                                                              name='d_loss1'))
            d_loss2 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_discriminator.logits,
                                                                              labels=tf.zeros_like(self.fake_discriminator.logits),
                                                                              name='d_loss2'))
            d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_fake_discriminator.logits,
                                                                              labels=tf.zeros_like(self.real_fake_discriminator.logits),
                                                                              name='d_loss3'))
            
            self.d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_discriminator.logits,
                                                                                  labels=tf.ones_like(self.fake_discriminator.logits),
                                                                                  name='g_loss'))
#             epsilon = tf.random_uniform([], 0.0, 1.0, dtype = tf.float32)
#             x_hat = epsilon * self.real_image + (1.0 - epsilon) * self.generator.outputs
#             x_hat_discriminator = Discriminator(x_hat, self.text_encoder.outputs, training_phase=True,
#                                               hparas=self.hparas, reuse=True)
#             gradients = tf.gradients(x_hat_discriminator.logits, x_hat)[0]
#             lamda = 10
#             gradient_penalty = lamda * tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1])) - 1.0)**2)
#             #gradient_penalty = lamda * tf.reduce_mean((tf.norm(gradients) - 1.0)**2)
#             self.d_loss += gradient_penalty   
    
    def _optimize(self):
        if self.train:
            with tf.variable_scope('learning_rate'):
                self.lr_var = tf.Variable(self.hparas['LR'], trainable=False)

            discriminator_optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=self.hparas['BETA'])
            generator_optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=self.hparas['BETA'])
            
#             gvs = discriminator_optimizer.compute_gradients(self.d_loss)
#             capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#             self.d_optim = discriminator_optimizer.apply_gradients(capped_gvs)
            
#             gvs = generator_optimizer.compute_gradients(self.g_loss)
#             capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#             self.g_optim = generator_optimizer.apply_gradients(capped_gvs)

            self.d_optim = discriminator_optimizer.minimize(self.d_loss, var_list=self.discrim_vars)
            self.g_optim = generator_optimizer.minimize(self.g_loss, var_list=self.generator_vars+self.text_encoder_vars)
        
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_loss, self.image_encoder_vars + self.text_encoder_vars), 10)
            optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=self.hparas['BETA'])
            self.rnn_optim = optimizer.apply_gradients(zip(grads, self.image_encoder_vars + self.text_encoder_vars))
        
    def training(self):
        
        for _epoch in range(self.hparas['N_EPOCH']):
            start_time = time.time()
            # Update learning rate
            if _epoch != 0 and (_epoch % self.hparas['DECAY_EVERY_EPOCH'] == 0):
                self.lr_decay = self.hparas['LR_DECAY'] ** (_epoch // self.hparas['DECAY_EVERY_EPOCH'])
                sess.run(tf.assign(self.lr_var, self.hparas['LR'] * self.lr_decay))
            
            n_batch_epoch = int(self.hparas['N_SAMPLE']/self.hparas['BATCH_SIZE'])
            for _step in range(n_batch_epoch):
                step_time = time.time()
                image_batch, caption_batch = self.sess.run(self.next_element)
                b_z = np.random.normal(loc=0.0, scale=1.0, 
                                       size=(self.hparas['BATCH_SIZE'], self.hparas['Z_DIM'])).astype(np.float32)
                if _epoch < 200:
                    self.encoder_error, _ = self.sess.run([self.rnn_loss, self.rnn_optim],
                                                         feed_dict={
                                                             self.real_image:image_batch,
                                                             self.caption:caption_batch
                                                         })
                else:
                    self.encoder_error = 0
                
                # update discriminator
                self.discriminator_error, _ = self.sess.run([self.d_loss, self.d_optim],
                                                           feed_dict={
                                                                self.real_image:image_batch,
                                                                self.caption:caption_batch,
                                                                self.z_noise:b_z})

                # update generate
                self.generator_error, _ = self.sess.run([self.g_loss, self.g_optim],
                                                       feed_dict={self.caption: caption_batch, self.z_noise : b_z})
                
                if _step%50==0:
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.3f, g_loss: %.3f, rnn_loss: %.3f"                             % (_epoch, self.hparas['N_EPOCH'], _step, n_batch_epoch, time.time() - step_time,
                               self.discriminator_error, self.generator_error, self.encoder_error))
            if _epoch != 0 and (_epoch+1)%5==0:
                self._save_checkpoint(_epoch)
                self._sample_visiualize(_epoch)
            
    def inference(self):
        for _iters in range(100):
            caption, idx = self.sess.run(self.next_element)
            z_seed = np.random.normal(loc=0.0, scale=1.0, size=(self.hparas['BATCH_SIZE'], self.hparas['Z_DIM'])).astype(np.float32)

            img_gen, rnn_out = self.sess.run([self.generate_image_net.outputs, self.text_embed.outputs],
                                             feed_dict={self.caption : caption, self.z_noise : z_seed})
            for i in range(self.hparas['BATCH_SIZE']):
                scipy.misc.imsave(self.inference_path+'/inference_{:04d}.png'.format(idx[i]), img_gen[i])
    def test_pred(self, caption):
        z_seed = np.random.normal(loc=0.0, scale=1.0, size=(self.hparas['BATCH_SIZE'], self.hparas['Z_DIM'])).astype(np.float32)
        text_embed = TextEncoder(self.caption, hparas=self.hparas, training_phase=False, reuse=True)
        generate_image_net = Generator(self.z_noise, text_embed.outputs, training_phase=False,
                                                hparas=self.hparas, reuse=True)
        img_gen, rnn_out = self.sess.run([generate_image_net.outputs, text_embed.outputs],
                                             feed_dict={self.caption : caption, self.z_noise : z_seed})
        print(img_gen.shape)
        for i in range(3):
            plt.imshow(img_gen[i])
                
    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())
    
    def _get_session(self):
        self.sess = tf.Session()
    
    def _get_saver(self):
        if self.train:
            self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars)
            self.cnn_saver = tf.train.Saver(var_list=self.image_encoder_vars)
            self.g_saver = tf.train.Saver(var_list=self.generator_vars)
            self.d_saver = tf.train.Saver(var_list=self.discrim_vars)
        else:
            self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars)
            self.g_saver = tf.train.Saver(var_list=self.generator_vars)
            
    def _sample_visiualize(self, epoch):
        ni = int(np.ceil(np.sqrt(self.hparas['BATCH_SIZE'])))
        sample_size = self.hparas['BATCH_SIZE']
        max_len = self.hparas['MAX_SEQ_LENGTH']
        
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, self.hparas['Z_DIM'])).astype(np.float32)
        sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."]*int(sample_size/ni) + ["this flower has petals that are yellow, white and purple and has dark lines"]*int(sample_size/ni) + ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + ["this flower has petals that are blue and white."] * int(sample_size/ni) + ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)

        for i, sent in enumerate(sample_sentence):
            sample_sentence[i] = sent2IdList(sent, max_len)
            
        img_gen, rnn_out = self.sess.run([self.generator.outputs, self.text_encoder.outputs],
                                         feed_dict={self.caption : sample_sentence, self.z_noise : sample_seed})
        save_images(img_gen, [ni, ni], self.sample_path+'/train_{:02d}.png'.format(epoch))
        
    def _get_var_with_name(self):
        t_vars = tf.trainable_variables()

        self.text_encoder_vars = [var for var in t_vars if 'rnn' in var.name]
        self.image_encoder_vars = [var for var in t_vars if 'cnn' in var.name]
        self.generator_vars = [var for var in t_vars if 'generator' in var.name]
        self.discrim_vars = [var for var in t_vars if 'discrim' in var.name]
    
    def _load_checkpoint(self, recover):
        if self.train:
            self.rnn_saver.restore(self.sess, self.ckpt_path+'rnn_model_'+str(recover)+'.ckpt')
            self.cnn_saver.restore(self.sess, self.ckpt_path+'cnn_model_'+str(recover)+'.ckpt')
            self.g_saver.restore(self.sess, self.ckpt_path+'g_model_'+str(recover)+'.ckpt')
            self.d_saver.restore(self.sess, self.ckpt_path+'d_model_'+str(recover)+'.ckpt')
        else:
            self.rnn_saver.restore(self.sess, self.ckpt_path+'rnn_model_'+str(recover)+'.ckpt')
            self.g_saver.restore(self.sess, self.ckpt_path+'g_model_'+str(recover)+'.ckpt')
        print('-----success restored checkpoint--------')
    
    def _save_checkpoint(self, epoch):
        self.rnn_saver.save(self.sess, self.ckpt_path+'rnn_model_'+str(epoch)+'.ckpt')
        self.cnn_saver.save(self.sess, self.ckpt_path+'cnn_model_'+str(epoch)+'.ckpt')
        self.g_saver.save(self.sess, self.ckpt_path+'g_model_'+str(epoch)+'.ckpt')
        self.d_saver.save(self.sess, self.ckpt_path+'d_model_'+str(epoch)+'.ckpt')
        print('-----success saved checkpoint--------')


# In[69]:


tf.reset_default_graph()
checkpoint_path = './checkpoint/'
inference_path = './inference'
gan = GAN(get_hparas(), training_phase=True, dataset_path=data_path, ckpt_path=checkpoint_path, inference_path=inference_path)
gan.training()


# In[70]:


def data_iterator_test(filenames, batch_size):
    data = pd.read_pickle(filenames)
    captions = data['Captions'].values
    caption = []
    for i in range(len(captions)):
        caption.append(captions[i])
    caption = np.asarray(caption)
    index = data['ID'].values
    index = np.asarray(index)
    
    dataset = tf.data.Dataset.from_tensor_slices((caption, index))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes
    
    return iterator, output_types, output_shapes


# In[71]:


tf.reset_default_graph()
iterator_train, types, shapes = data_iterator_test(data_path+'/testData.pkl', 64)
iter_initializer = iterator_train.initializer
next_element = iterator_train.get_next()

with tf.Session(config = config) as sess:
    sess.run(iterator_train.initializer)
    next_element = iterator_train.get_next()
    caption, idex = sess.run(next_element)


# In[102]:


tf.reset_default_graph()
gan = GAN(get_hparas(), training_phase=False, dataset_path=data_path, ckpt_path=checkpoint_path, inference_path=inference_path, recover=99)
img = gan.inference()


# In[103]:


BATCH_SIZE = 64
iterator_train, types, shapes = data_iterator(data_path+'/text2ImgData.pkl', BATCH_SIZE, training_data_generator)
iter_initializer = iterator_train.initializer
next_element = iterator_train.get_next()

with tf.Session(config = config) as sess:
    sess.run(iterator_train.initializer)
    next_element = iterator_train.get_next()
    image, text = sess.run(next_element)
print(text.shape)
plt.imshow(image[0])
gan.test_pred(text)


# In[20]:





# In[19]:


os.system('cd testing && python ./inception_score.py ../inference ../socre.csv')


# In[ ]:




