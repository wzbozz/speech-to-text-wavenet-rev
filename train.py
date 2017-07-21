import sugartensor as tf
from data import SpeechCorpus, voca_size
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())

# mfcc feature of audio
labels = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
# target sentence label
inputs = tf.split(data.label, tf.sg_gpus(), axis=0)
# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_, 0).sg_int().sg_sum(axis=1))

# parallel loss tower
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logit = get_logit(opt.input[opt.gpu_index].sg_float().sg_expand_dims(), voca_size=voca_size)
    
    # CTC loss
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])
    # mse的问题是向量长度要相等，若生成序列不等长就会出错，截断？
    #return logit.sg_mse(target=labels[0]).sg_sum(axis=2).sg_sum(axis=1)

#
# train
#
loss = get_loss(input=inputs, target=labels, seq_len=seq_len)
tf.sg_train(lr=0.0001, loss=loss,
            ep_size=data.num_batch, max_ep=50)
