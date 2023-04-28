import tensorflow as tf
import tensorflow.compat.v1 as tf


class Config:
	language = 'fr_en'  # zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'
	epochs = 200
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	k = 125  # number of negative samples for each positive one
	seed = 3  # 30% of seeds
	can_len = 5
