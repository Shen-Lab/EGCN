from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Add, Concatenate, Conv2D, Reshape, Lambda, Dropout
from tensorflow.keras.layers import Multiply, TimeDistributed, LeakyReLU, Layer, BatchNormalization
import tensorflow.keras.backend as K
import numpy as np
import model0_residue_based
   
def __data_generation(start, end, data_path):
      c1 =[]
      c2 =[]
      c3 =[]
      c4 =[]
      c5 =[]
      for i in range(start, end):

          c1.append(np.load(data_path+"/node_rec_"+str(i+1)+".npy"))
          c2.append(np.load(data_path+"/node_lig_"+str(i+1)+".npy"))
          c3.append(np.load(data_path+"/intra_rec_"+str(i+1)+".npy"))
          c4.append(np.load(data_path+"/intra_lig_"+str(i+1)+".npy"))
          c5.append(np.load(data_path+"/inter"+str(i+1)+".npy"))
          
 
      return [np.array(c1),np.array(c2),np.array(c3),np.array(c4), np.array(c5)]

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", "./scores", "Path for saving predicted scores")
flags.DEFINE_string("model_path", '../trained_models/checkpoints_lr0.01round70_03-484.86.hdf5', 'path to the trained model')
flags.DEFINE_string("data_path", '../../data/7_resi_final/', 'data path')
flags.DEFINE_integer("nums", 100, "number of decoys within this target")

def main():
      
      #flex_list= np.loadtxt(FLAGS.test_list, dtype='str')
      #flex_num = np.loadtxt("../../data/statistic/flexible_list_num", dtype='int')
      

      nums = FLAGS.nums
      model = model0_residue_based.my_model()    
      pretrained_model = model.main_model()
      pretrained_model.load_weights(FLAGS.model_path)
      eval_model = model.eval_model(pretrained_model)
      
      for i in range(1):
         start=0
         batch_size=16
         y=np.zeros((nums,))
         while(1):
           if(start+batch_size <= nums):
               x = __data_generation(start, start+batch_size, FLAGS.data_path)
               y[start:start+batch_size] = np.reshape(eval_model.predict(x), (-1,))+60
           else:
               x = __data_generation(start, nums, FLAGS.data_path)
               y[start:nums]     =  np.reshape(eval_model.predict(x), (-1,))+60
               break
           print ("ite/total ite: %d/%d"  %(start, nums ) )
           start+=batch_size

         np.savetxt(FLAGS.save_path, y)
          
         
      


if __name__ == "__main__":
   main()


         





