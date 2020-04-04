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

class DataGenerator(tf.keras.utils.Sequence):
   'Generate Data'
   def __init__(self, protein_list, data_path, total_sample, label_path, batch_size=16):
      self.batch_size = batch_size
      self.total_sample = total_sample
      self.on_epoch_end()
      self.protein_list= protein_list
      self.data_path = data_path
      self.label_path = label_path

   def __len__(self):
      return int(self.total_sample/self.batch_size)
   
   def __getitem__(self, idx):
      indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
      
      x,y = self.__data_generation(indexes)
      return x,y
   
   def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.total_sample)
      np.random.shuffle(self.indexes)

   
   def __data_generation(self, indexes):
      c1 =[]
      c2 =[]
      c3 =[]
      c4 =[]
      c5 =[]
      c6 =[]
      c7 =[]
      c8 =[]
      c9=[]
      cy=[]
      label = np.loadtxt(self.label_path)      
      for i in indexes:
          protein_idx = int(i/1000)
          decoy_idx   = i%1000+1

          c1.append(np.load(self.data_path+"/"+self.protein_list[protein_idx]+"/node_rec_"+str(decoy_idx)+".npy"))
          c2.append(np.load(self.data_path+"/"+self.protein_list[protein_idx]+"/node_lig_"+str(decoy_idx)+".npy"))
          c3.append(np.load(self.data_path+"/"+self.protein_list[protein_idx]+"/intra_rec_"+str(decoy_idx)+".npy"))
          c4.append(np.load(self.data_path+"/"+self.protein_list[protein_idx]+"/intra_lig_"+str(decoy_idx)+".npy"))
          c5.append(np.load(self.data_path+"/"+self.protein_list[protein_idx]+"/inter"+str(decoy_idx)+".npy"))
          c6.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/node_rec.npy"))
          c7.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/node_lig.npy"))
          c8.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/intra_rec.npy"))
          c9.append(np.load("../data/unbound_pdb/"+self.protein_list[protein_idx]+"/intra_lig.npy"))
          cy.append(label[i])
          
          
 
      return [np.array(c1),np.array(c2),np.array(c3),np.array(c4),np.array(c5),np.array(c6),np.array(c7),np.array(c8),np.array(c9)],np.array(cy)

class LossHistory(keras.callbacks.Callback):
    def __init__(self, lr):
        self.lr = lr
    def on_train_begin(self, logs={}):
        self.losses = []
       
    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        with open(FLAGS.save_path+"/lr"+str(self.lr)+"_multi"+str(FLAGS.multi)+"_loss", "a") as f:
             f.write("%.3f\n" %(logs.get('loss')))

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", "./log_model0_residue_based_v2/", "Path for saved model")
flags.DEFINE_integer("epochs", 10 , "Number of training epochs")
flags.DEFINE_string("train_list", "../data/statistic/train_list" , "training protein list")
flags.DEFINE_string("val_list",   "../data/statistic/train_list" , "validation protein list")
flags.DEFINE_float("alpha",  1.5 , "hyperparameter alpha in the label")
flags.DEFINE_float("q",   0.5 , "hyperparameter q in the label")
flags.DEFINE_string("train_round",  '1' , "train_round")
flags.DEFINE_float("lr",   0.001 , "learning rate")
flags.DEFINE_bool("multi",  False , "multi_processing")
flags.DEFINE_string("resume_train", 'None', 'give path if continuing training')
flags.DEFINE_string("resume_epoch", '0', 'resume train epoch')

def label(alpha, q, list1):
    label = [] 
    kd_name = np.loadtxt("../data/statistic/kd_list", dtype='str')
    kd_zero = np.loadtxt("../data/statistic/kd_zero")
    for i in range(len(list1)):
        for j in range(1000):
           irmsd = np.loadtxt("../data/3_decoys/"+list1[i]+"_"+str(j+1)+"/energy_label.dat")
        
           kd_value = kd_zero[np.where(kd_name == list1[i])[0][0]]
           rt=8.3
           label.append(rt* np.log( kd_value) +  rt * alpha * irmsd[0][8]**q )

        print ("calculating_label: protein", i+1) 
    return label
        



def main():
      
      data_list = np.loadtxt("../data/statistic/train_list", dtype='str')
      val_list = data_list[40:44]
      train_list = data_list[0:40]
 
      #label_train =label(FLAGS.alpha, FLAGS.q, train_list)
      #label_val =label(FLAGS.alpha, FLAGS.q, val_list)
      #np.savetxt("log_model0_residue_based/label_train_"+FLAGS.train_round, label_train)
      #np.savetxt("log_model0_residue_based/label_val_"+FLAGS.train_round, label_val)


      train_generator = DataGenerator(train_list, '../data/7_resi_final/', len(train_list)*1000 , FLAGS.save_path+"label_train_"+FLAGS.train_round)
      val_generator   = DataGenerator(val_list, '../data/7_resi_final/',   len(val_list)*1000   , FLAGS.save_path+"label_val_"+FLAGS.train_round)
      
      if FLAGS.resume_train!= 'None':
         
         model = model0_residue_based.my_model()
         custom_ob ={'matmul': model.matmul,
             'Self_attn': model0_residue_based.Self_attn
         }
         model1=load_model(FLAGS.resume_train, custom_objects= custom_ob)
         '''
         model = model0_residue_based.my_model()    
         model1 = model.main_model()
         model1.load_weights(FLAGS.resume_train)
         adam = tf.keras.optimizers.Adam(lr=FLAGS.lr)
         model1.compile(optimizer=adam, loss='mean_squared_error')        
         '''

         print (K.eval(model1.optimizer.lr), "learning rate")
         
      else:
     
         model = model0_residue_based.my_model()    
         model1 = model.main_model()
         adam = tf.keras.optimizers.Adam(lr=FLAGS.lr)
         model1.compile(optimizer=adam, loss='mean_squared_error')
         print (K.eval(model1.optimizer.lr), "learning rate") 
      
      checkpath = FLAGS.save_path+"/checkpoints_lr"+str(K.eval(model1.optimizer.lr))+"round"+FLAGS.resume_epoch+"_{epoch:02d}-{val_loss:.2f}.hdf5"
      checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,mode='auto') 
     
      history = LossHistory(K.eval(model1.optimizer.lr))

      model1.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    callbacks= [checkpoints, history],
                    use_multiprocessing=bool(FLAGS.multi),
                     workers=2,
                     epochs=FLAGS.epochs)
      
     
      #model1.save(FLAGS.save_path+"/trained_modelresi0.h5") 


if __name__ == "__main__":
   main()


         





