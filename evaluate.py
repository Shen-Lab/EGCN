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
   
def __data_generation(protein, start, end, data_path):
      c1 =[]
      c2 =[]
      c3 =[]
      c4 =[]
      c5 =[]
      c6 =[]
      c7 =[]
      c8 =[]
      c9=[]
      #cy=[]
      #label = np.loadtxt(self.label_path)      
      for i in range(start, end):

          c1.append(np.load(data_path+"/"+protein+"/node_rec_"+str(i+1)+".npy"))
          c2.append(np.load(data_path+"/"+protein+"/node_lig_"+str(i+1)+".npy"))
          c3.append(np.load(data_path+"/"+protein+"/intra_rec_"+str(i+1)+".npy"))
          c4.append(np.load(data_path+"/"+protein+"/intra_lig_"+str(i+1)+".npy"))
          c5.append(np.load(data_path+"/"+protein+"/inter"+str(i+1)+".npy"))
          #c6.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/node_rec.npy"))
          #c7.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/node_lig.npy"))
          #c8.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/intra_rec.npy"))
          #c9.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/intra_lig.npy"))
          #cy.append(label[i])
          
          
 
      return [np.array(c1),np.array(c2),np.array(c3),np.array(c4), np.array(c5)]

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", "./eval_score/", "Path for saved model")
flags.DEFINE_integer("epochs", 10 , "Number of training epochs")
flags.DEFINE_string("test_list", "../data/statistic/flexible_list" , "testing protein list")
flags.DEFINE_string("val_list",   "../data/statistic/train_list" , "validation protein list")
flags.DEFINE_float("alpha",  1.5 , "hyperparameter alpha in the label")
flags.DEFINE_float("q",   0.5 , "hyperparameter q in the label")
flags.DEFINE_string("train_round",  '1' , "train_round")
flags.DEFINE_float("lr",   0.001 , "learning rate")
flags.DEFINE_bool("multi",  False , "multi_processing")
flags.DEFINE_string("resume_train", 'None', 'give path if continuing training')
flags.DEFINE_string("data_path", '../data/7_resi_final/', 'data path')


def main():
      
      flex_list= np.loadtxt(FLAGS.test_list, dtype='str')
      flex_num = np.loadtxt("../data/statistic/flexible_list_num", dtype='int')
      

      
      model = model0_residue_based.my_model()    
      pretrained_model = model.main_model()
      pretrained_model.load_weights(FLAGS.resume_train)
      eval_model = model.eval_model(pretrained_model)
      
      for i in range(4, len(flex_list)):
         start=0
         batch_size=16
         y=np.zeros((flex_num[i],))
         while(1):
           if(start+batch_size <= flex_num[i]):
               x = __data_generation(flex_list[i], start, start+batch_size, FLAGS.data_path)
               y[start:start+batch_size] = np.reshape(eval_model.predict(x), (-1,))
           else:
               x = __data_generation(flex_list[i], start, flex_num[i], FLAGS.data_path)
               y[start:flex_num[i]]     =  np.reshape(eval_model.predict(x), (-1,))
               break
           print (start, y[start], flex_list[i]) 
           start+=batch_size

         np.savetxt(FLAGS.save_path+"/score_"+flex_list[i], y)
          
         
      


if __name__ == "__main__":
   main()


         





