#!/usr/bin/env python

""""
Example Usage:
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True

Papers:
DRAW RNN for image generation http://arxiv.org/pdf/1502.04623v2.pdf (Deepmind)
Fast Weights https://arxiv.org/abs/1610.06258 (Hinton et. al.)

Based on implementations by:
Eric Jang http://blog.evjang.com/2016/06/understanding-and-implementing.html
Goku Mohandas (gokumd@gmail.com) https://theneuralperspective.com/2016/12/04/implementation-of-using-fast-weights-to-attend-to-the-recent-past/

Hybrid implementation by:
Matt Shaffer (matt@discovermatt.com)

"""


import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import time
import sys

from fastweights import (
    fast_weights_encoding,
)
from op_utils import (
    print_wipe,
    backspace,
)

## OPTIONS ##
tf.flags.DEFINE_string("data_dir", "./tmp/data", "")
tf.flags.DEFINE_string("save_model_name", "fastdraw", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
tf.flags.DEFINE_boolean("fast_enc",True, "enable fast weights for encoder")
tf.flags.DEFINE_boolean("fast_dec",True, "enable fast weights for decoder")
tf.flags.DEFINE_boolean("fast_weights",True, "enable fast weights for both encoder decoder")
tf.flags.DEFINE_boolean("print_variables", False, "print variable shapes")
tf.flags.DEFINE_boolean("verbose", False, "verbose option")
tf.flags.DEFINE_boolean("train", True, "training mode")

## SHARED VARIABLES ##
tf.flags.DEFINE_integer("T", 10, "MNIST generation sequence length")# [default = 10]
tf.flags.DEFINE_integer("batch_size", 100, "Training minibatch size") # [default = 100]
tf.flags.DEFINE_integer("train_iters", 100, "Training minibatch size") # [default = 10000]

## FAST WEIGHTS PARAMETERS ##
tf.flags.DEFINE_integer("input_dim", 1, "Input dimensions (currently disabled)")  # [1, 9]
tf.flags.DEFINE_float("slow_init", 0.05, "Slow weights initialization scaling") # [default = 0.05] (See Hinton's video @ 21:20)
tf.flags.DEFINE_integer("num_hidden_units", 10,  "Hidden units") # [default = 100] [50,100,200]
tf.flags.DEFINE_float("decay_lambda", 0.9,  "decay lambda value") # [default = 0.9] [0.9, 0.95]
tf.flags.DEFINE_float("rate_eta", 0.5,  "learning rate(eta)") # [default = 0.5]
tf.flags.DEFINE_integer("S", 3,  "inner loops where h(t+1) is transformed into h_S(t+1)") # [default = 3]

FLAGS = tf.flags.FLAGS
MSG = ""

save_model_name = FLAGS.save_model_name + ".ckpt"
save_meta = save_model_name + ".meta"
t0 = time.time()

## MODEL PARAMETERS ##
print_wipe("Loading Model Parameters...")
A,B = 28,28 # [default = 28,28] image width,height
img_size = B*A # canvas size
enc_size = 256 # [default = 256] number of hidden units / output size in LSTM
dec_size = 256
read_n = 5 # [default = 5] read glimpse grid width/height
write_n = 5 # [default = 5] write glimpse grid width/height
z_size = 10 # [default = 10] QSampler output size

read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n if FLAGS.write_attn else img_size
T = FLAGS.T # [default = 10] MNIST generation sequence length
batch_size = FLAGS.batch_size # [default = 100] training minibatch size
train_iters = FLAGS.train_iters # [default = 10000]

learning_rate = 1e-3 # [default = 1e-3] learning rate for optimizer
eps = 1e-8 # [default = 1e-8] epsilon for numerical stability (See Jang's blog post)

print_wipe("Loading Model...")

## BUILD MODEL ##
DO_SHARE = None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
e = tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, state_is_tuple=True) # decoder Op
global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, dtype=tf.int32) # Log global steps

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ##
def read_no_attn(x,x_hat,h_dec_prev):
    return tf.concat(1,[x,x_hat])

def read_attn(x,x_hat,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)

    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])

    x = filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)

    # Fast weights for encoder
    # We will create a fast weights matrix for each output class, instead of creating a matrix with n*class dimensions
    fw_x = fast_weights(x, int(x.get_shape()[1]), "fast_weights_enc")

    x_hat = filter_img(x_hat,Fx,Fy,gamma,read_n)

    return tf.concat(1,[x, x_hat, fw_x]) # concat along feature axis

read = read_attn if FLAGS.read_attn else read_no_attn

def fast_weights_payload(x, input_size, state):
    return fast_weights_encoding(x, input_size, state, FLAGS, DO_SHARE)

fast_weights = fast_weights_payload if FLAGS.fast_weights else []

## ENCODE ##
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder", reuse=DO_SHARE):
        return lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ##
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(input, state)

## WRITER ##
def write_no_attn(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,write_size) # batch x (write_n*write_n)
    N=write_n
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn_window("write",h_dec,write_n)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.batch_matmul(Fyt,tf.batch_matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
    #gamma=tf.tile(gamma,[1,B*A])
    return wr*tf.reshape(1.0/gamma,[-1,1])

write=write_attn if FLAGS.write_attn else write_no_attn

## STATE VARIABLES ##

cs=[0]*T # sequence of canvases
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ##

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,img_size)) if t == 0 else cs[t-1]

    x_hat=x-tf.sigmoid(c_prev) # error image
    print('x shape: {0}'.format(x.get_shape())) if FLAGS.print_variables else 0

    r = read(x, x_hat, h_dec_prev)
    print("r shape: {0}".format(r.get_shape())) if FLAGS.print_variables else 0

    h_enc,enc_state=encode(enc_state,tf.concat(1,[r,h_dec_prev]))
    z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)
    print("z shape: {0}".format(z.get_shape())) if FLAGS.print_variables else 0

    # fast weights for decoder
    fw_dec = fast_weights(z, int(z.get_shape()[1]), "fast_weights_dec")

    h_dec, dec_state = decode(dec_state, tf.concat(1, [z, fw_dec]))
    cs[t]=c_prev+write(h_dec) # store results

    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

print_wipe("Loading optimization functions...")

## LOSS FUNCTION ##

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

## OPTIMIZER ##

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op = optimizer.apply_gradients(grads, global_step=global_step)

msg = "Loading training data...\n"
print_wipe(msg)
## RUN TRAINING ##
if FLAGS.train:
    data_directory = "./mnist"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
    #test_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test  # binarized (0-1) mnist data
    #validation_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).validation  # binarized (0-1) mnist data
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(train_data.labels)))
    #print("- Test-set:\t\t{}".format(len(test_data.labels)))
    #print("- Validation-set:\t{}".format(len(validation_data.labels)))
    #test_data.cls = np.argmax(test_data.labels, axis=1)
    #validation_data.cls = np.argmax(validation_data.labels, axis=1)
    print("Data Loaded")

    fetches=[]
    fetches.extend([Lx,Lz,train_op])
    Lxs=[0]*train_iters
    Lzs=[0]*train_iters

    ## Check model parameters ##
    if FLAGS.print_variables:
        for v in tf.all_variables():
            print("{0} : {1}".format(v.name, v.get_shape()))

    ## Start or resume Tensorflow session ##
    saver = tf.train.Saver(tf.global_variables())  # saves variables learned during training
    with tf.Session() as sess:
        try:
            if os.path.exists(FLAGS.data_dir):
                msg = "Attempting to restore prior checkpoint ..."
                print(msg)
                ckpt_last_path = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.data_dir)
                backspace(len(msg))
                msg = "Save path: {0}".format(ckpt_last_path)
                print(msg)
                saver.restore(sess, ckpt_last_path)
                print("Checkpoints restored successfully")
                if FLAGS.print_variables:
                    all_vars = tf.trainable_variables()
                    for v in all_vars:
                        print(v.name)
        except Exception as e:
            print("Unable to locate or restore existing model: {0}".format(e))
            print("Creating new model.")
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.train.SummaryWriter(FLAGS.data_dir, sess.graph)
        t1 = time.time()
        completed_steps = sess.run(global_step)
        print("Training from iteration: {0}".format(completed_steps))
        for i in range(train_iters-completed_steps):
            xtrain,_= train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
            feed_dict = {x:xtrain}
            i_global, results = sess.run([global_step, fetches],feed_dict)
            Lxs[i], Lzs[i],_ = results
            if (i_global % 100 == 0) and not (i_global % 500 == 0):
                msg = "iter:{0}  Lx: {1:.6f} Lz: {2:.6f} | {3:.1f}sec / {4:.0f}sec (iter/elapsed)"
                print(msg.format(i_global, Lxs[i], Lzs[i], time.time()-t1, time.time()-t0))
                t1 = time.time()
            if i_global % 500 == 0:
                msg = "iter:{0}  Lx: {1:.6f} Lz: {2:.6f} | {3:.1f}sec / {4:.0f}sec (iter/elapsed)"
                print(msg.format(i_global, Lxs[i], Lzs[i], time.time()-t1, time.time()-t0))
                t2 = time.time()
                if not os.path.isdir(FLAGS.data_dir):
                    os.makedirs(FLAGS.data_dir)
                checkpoint_path = os.path.join(FLAGS.data_dir, "%s.ckpt" % FLAGS.save_model_name)
                saver.save(sess, checkpoint_path, global_step = global_step)
                print("Checkpoint saved {0:.0f} sec".format(time.time()-t2))
                t1 = time.time()

        ## TRAINING FINISHED ##

        canvases=sess.run(cs,feed_dict) # generate some examples
        canvases=np.array(canvases) # T x batch x img_size

        out_file=os.path.join(FLAGS.data_dir,"fastdraw_data.npy")
        np.save(out_file,[canvases,Lxs,Lzs])
        print("Outputs saved in file: %s" % out_file)

        ckpt_file=os.path.join(FLAGS.data_dir, save_model_name)
        print("Model saved in file: %s" % saver.save(sess,ckpt_file))

        print('Session End. Training Completed in {0:.1f} seconds. '.format(time.time() - t0))
