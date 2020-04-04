# model0_residue_based:  The v0 residue-based model

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Add, Concatenate, Conv2D, Reshape, Lambda, Dropout
from tensorflow.keras.layers import Multiply, TimeDistributed, LeakyReLU, Layer, BatchNormalization, Permute
import tensorflow.keras.backend as K
import numpy as np

class Self_attn(Layer):

	def __init__(self, attn_length, **kwargs):
		self.attn_length = attn_length
		super(Self_attn, self).__init__(**kwargs)

	def build(self, input_shape):           # input shape (None, 1500, # filters in the last GCN layer)
		# Create a trainable weight variable for this layer.
		#print (input_shape, type(input_shape.as_list()[2]), type(self.attn_length))
		self.W = self.add_weight(name='W', shape=(input_shape.as_list()[2],self.attn_length),initializer='random_uniform',trainable=True)
		self.b = self.add_weight(name='b', shape=(self.attn_length,),initializer='random_uniform',trainable=True)
		self.V = self.add_weight(name='V', shape=(self.attn_length,1), initializer='random_uniform', trainable=True)
		super(Self_attn, self).build(input_shape)  # Be sure to call this at the end

	def get_config(self):
		config = {'attn_length': self.attn_length}
		base_config = super(Self_attn, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def call(self, x):
		U = tf.tanh(tf.tensordot(x, self.W, [[2], [0]]) + self.b)
		UV = tf.tensordot(U, self.V, [[2], [0]])
		UV = tf.squeeze(UV, axis=-1)
		p = tf.nn.softmax(UV)

		Attn = tf.reduce_sum(x*tf.expand_dims(p, -1), 1)

		return Attn

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

class my_model():

	def __init__(self, node_feature_length = 4, edge_feature_length = 11, residue_num=500):
		self.node_feature_length = node_feature_length
		self.edge_feature_length = edge_feature_length
		self.residue_num = residue_num

	def config(self, Wat_length=10, conv1_depth=4, conv2_depth=4, conv3_depth=4 ):
		self.Wat_length = Wat_length
		self.conv1_depth = conv1_depth
		self.conv2_depth = conv2_depth
		self.conv3_depth = conv3_depth


	def matmul(self, var,  transpose_a, transpose_b):
		return tf.linalg.matmul(var[0], var[1], transpose_a=transpose_a, transpose_b=transpose_b)

	def GCN_layer(self, p1_node_layershape, p2_node_layershape, edge_layershape, output_dim):
		print (p1_node_layershape)
		p1_node_layer =  Input(shape=(p1_node_layershape.as_list()[1], p1_node_layershape.as_list()[2]) , dtype='float32')
		p2_node_layer =  Input(shape=(p2_node_layershape.as_list()[1], p2_node_layershape.as_list()[2]) , dtype='float32')
		edge_layer =  Input(shape=(edge_layershape.as_list()[1], edge_layershape.as_list()[2],edge_layershape.as_list()[3]) , dtype='float32')

		# ------------------------------------first around of convolution
		print ('p1_node_layer', p1_node_layer.shape)
		Wx_dense = TimeDistributed(Dense(self.Wat_length*output_dim*self.edge_feature_length, use_bias=False))



		WatXp1_pre = Wx_dense(p1_node_layer)
		WatXp2_pre = Wx_dense(p2_node_layer)

		WatXp1 = Reshape((output_dim, self.edge_feature_length, self.residue_num, self.Wat_length ))(WatXp1_pre)
		WatXp2 = Reshape((output_dim, self.edge_feature_length, self.residue_num, self.Wat_length))(WatXp2_pre)

		print ('WatXp1', WatXp1_pre.shape, output_dim)
		# shape = (-1, -1)
		WatXp1_matmul_WatXp2 = Lambda(self.matmul, arguments={'transpose_a' : False, 'transpose_b':True})((WatXp1, WatXp2))

		print (WatXp1_matmul_WatXp2.shape, edge_layer.shape)

		edge_layer_reshape = Reshape((self.edge_feature_length, self.residue_num, self.residue_num))(edge_layer)

		WatXp1_matmul_WatXp2_multiply_edge = Multiply()([WatXp1_matmul_WatXp2, edge_layer_reshape])

		Conv_output_pre = Lambda(lambda x: K.sum(x, axis=-1))(WatXp1_matmul_WatXp2_multiply_edge)
		Conv_output_pre2 = Lambda(lambda x: K.sum(x, axis=2))(Conv_output_pre)
		Conv_output_pre3 = Reshape((self.residue_num, output_dim))(Conv_output_pre2)

		Conv_output_pre4 = LeakyReLU(alpha=0.3)(Conv_output_pre3)
		Conv_output = BatchNormalization()(Conv_output_pre4)
		
		return Model(inputs=[p1_node_layer, p2_node_layer, edge_layer], outputs=Conv_output)


	def GCN_module_intra(self):
		p1_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		p2_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		edge_layer    =  Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32') 
		
		GCN_model = self.GCN_layer(p1_node_layer.shape, p2_node_layer.shape, edge_layer.shape, self.conv1_depth)
		GCN_output_1 = GCN_model([p1_node_layer, p2_node_layer, edge_layer])

		GCN_model = self.GCN_layer(GCN_output_1.shape, GCN_output_1.shape, edge_layer.shape, self.conv2_depth)
		GCN_output_2 = GCN_model([GCN_output_1, GCN_output_1, edge_layer])

		GCN_model = self.GCN_layer(GCN_output_2.shape, GCN_output_2.shape, edge_layer.shape, self.conv3_depth)
		GCN_output_3 = GCN_model([GCN_output_2, GCN_output_2, edge_layer])

		return Model(inputs=[p1_node_layer, p2_node_layer, edge_layer], outputs=GCN_output_3)

	def GCN_module_inter(self):
		p1_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		p2_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		edge_layer    =  Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		
		edge_layer_transpose =  Permute((2,1,3))(edge_layer)
		
		GCN_model = self.GCN_layer(p1_node_layer.shape, p2_node_layer.shape, edge_layer.shape, self.conv1_depth)	
		GCN_output_1a = GCN_model([p1_node_layer, p2_node_layer, edge_layer])
		GCN_output_1b = GCN_model([p2_node_layer, p1_node_layer, edge_layer_transpose])
		
		GCN_model = self.GCN_layer(GCN_output_1a.shape, GCN_output_1b.shape, edge_layer.shape, self.conv2_depth)
		GCN_output_2a = GCN_model([GCN_output_1a,  GCN_output_1b, edge_layer])
		GCN_output_2b = GCN_model([GCN_output_1b,  GCN_output_1a, edge_layer_transpose])

		GCN_model = self.GCN_layer(GCN_output_2a.shape, GCN_output_2b.shape, edge_layer.shape, self.conv3_depth)
		GCN_output_3a = GCN_model([GCN_output_2a,  GCN_output_2b, edge_layer])
		GCN_output_3b = GCN_model([GCN_output_2b,  GCN_output_2a, edge_layer_transpose])	
		

		return Model(inputs=[p1_node_layer, p2_node_layer, edge_layer], outputs=[GCN_output_3a, GCN_output_3b])


	def multi_head_attention(self, inputs):


		print ('before_attention_shape', inputs.shape)
		
		#   we choose 5-heads self-attention here
		atten_out_head1 = Self_attn(attn_length=10)(inputs)
		atten_out_head2 = Self_attn(attn_length=10)(inputs)
		atten_out_head3 = Self_attn(attn_length=10)(inputs)
		atten_out_head4 = Self_attn(attn_length=10)(inputs)
		atten_out_head5 = Self_attn(attn_length=10)(inputs)

		atten_out = Concatenate(axis=-1)([atten_out_head1, atten_out_head2, atten_out_head3, atten_out_head4, atten_out_head5])

		print (atten_out.shape)

		return atten_out


	def energy_module(self, name):
		p1_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		p2_node_layer =  Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		edge_layer    =  Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32') 
		
		if 'intra' in name:	
			GCN_part = self.GCN_module_intra()
			GCN_out = GCN_part([p1_node_layer, p2_node_layer, edge_layer])
			GCN_out = Concatenate(axis=-2)([GCN_out, GCN_out])
		else:
			GCN_part = self.GCN_module_inter()
			GCN_out_list = GCN_part([p1_node_layer, p2_node_layer, edge_layer])
			GCN_out = Concatenate(axis=-2)(GCN_out_list)
		print ('GCN_out', GCN_out)
		
	
		#  multi_head_attention_part
		attn_out = self.multi_head_attention(GCN_out)

		#  fc part

		fc1 = Dense(64, kernel_initializer='glorot_uniform')(attn_out)
		fc1 = LeakyReLU(alpha=0.1)(fc1)
		drop1 = Dropout(0.2)(fc1)
		fc2 = Dense(32, kernel_initializer='glorot_uniform')(drop1)
		fc2 = LeakyReLU(alpha=0.1)(fc2)
		drop2 = Dropout(0.2)(fc2)
		score = Dense(1, kernel_initializer='glorot_uniform')(drop2)


		return Model(inputs=[p1_node_layer, p2_node_layer, edge_layer],
			outputs=score, name=name)


	def eval_model(self, pretrained_model):
		rec_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		lig_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		rec_intra_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		lig_intra_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		inter_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')

		intra = pretrained_model.get_layer("intra_model")
		inter = pretrained_model.get_layer("inter_model")
		score_rec_intra   = intra([rec_node_feature, rec_node_feature, rec_intra_edge_feature])
		score_lig_intra   = intra([lig_node_feature, lig_node_feature, lig_intra_edge_feature])
		score_inter       = inter([rec_node_feature, lig_node_feature, inter_edge_feature])

		def score_get(v):
                    return v[0]+v[1]+v[2]
		score = Lambda(score_get)([score_rec_intra, score_lig_intra, score_inter])

		eval_model = Model(inputs=[rec_node_feature, lig_node_feature, rec_intra_edge_feature, lig_intra_edge_feature, inter_edge_feature],
			outputs = score)

		return eval_model
	def main_model(self):

		self.config()

		# --------------real input
		rec_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		lig_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		rec_intra_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		lig_intra_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		inter_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		unbound_rec_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		unbound_lig_edge_feature = Input(shape=(self.residue_num, self.residue_num, self.edge_feature_length), dtype='float32')
		unbound_rec_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')
		unbound_lig_node_feature = Input(shape=(self.residue_num, self.node_feature_length), dtype='float32')

		intra_energy_module = self.energy_module('intra_model')
		inter_energy_module  = self.energy_module('inter_model')

		score_rec_intra   = intra_energy_module([rec_node_feature, rec_node_feature, rec_intra_edge_feature])
		score_lig_intra   = intra_energy_module([lig_node_feature, lig_node_feature, lig_intra_edge_feature])
		score_rec_unbound = intra_energy_module([unbound_rec_node_feature, unbound_rec_node_feature, unbound_rec_edge_feature])
		score_lig_unbound = intra_energy_module([unbound_lig_node_feature, unbound_lig_node_feature, unbound_lig_edge_feature])
		score_inter       = inter_energy_module([rec_node_feature, lig_node_feature, inter_edge_feature])


		def score_get(v):
			return v[0]+v[1]+v[2]-v[3]-v[4]
		score = Lambda(score_get)([score_rec_intra, score_lig_intra, score_inter, score_rec_unbound, score_lig_unbound])



		Final_GCN = Model(inputs=[rec_node_feature, lig_node_feature, rec_intra_edge_feature, lig_intra_edge_feature, inter_edge_feature, 
			unbound_rec_node_feature, unbound_lig_node_feature, unbound_rec_edge_feature, unbound_lig_edge_feature],
			outputs=score)

		Final_GCN.summary()
		print(Final_GCN.count_params())
		
		return Final_GCN

if __name__ == "__main__":
    mymodel1 = my_model()
    mymodel1.main_model()




	
