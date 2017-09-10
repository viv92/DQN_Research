import gym
from random import randint
env = gym.make('MountainCar-v0')

import tensorflow as tf
import argparse
import sys

# ------------------------------------hyperparams----------------------------------# 
alpha = 0.75 #learning rate
df = 0.99 #discount factor 
explore_count = 0 
explore_limit = 50
freerun_maxsteps = 1000
rm_reset_count = -1
max_samples = 1
max_iters = 1
rm_size = 100000
rm_full_flag = 0
tim = 0 #global time

#replay memory
rm_present_state = [[0,0]]*rm_size
rm_action = [0]*rm_size
rm_reward = [0]*rm_size
rm_next_state = [[0,0]]*rm_size

#-----------------------------------------Neural net helper methods--------------------------------#

#get interactive session
sess = tf.InteractiveSession()

FLAGS = None

#random weights initialisation function
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#bias initialisation function
def bias_variable(Shape):
	initial = tf.constant(0.1, shape=Shape)
	return tf.Variable(initial)

#convolution function
def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#-------------------------------------Neural net---------------------------------------------#
#placeholder for input images
x = tf.placeholder(tf.float32, shape=[2])

#placeholder for correct labels
y_ = tf.placeholder(tf.float32, shape=[None,3])

#reshape input to apply convolution
x_obs = tf.reshape(x, [-1, 1, 2, 1])

#first convolution layer params
W_conv1 = weight_variable([1, 1, 1, 5]) #input_channel = 1, output_channel = 5
b_conv1 = bias_variable([5])

#apply first convolution (no pooling)
h_conv1 = tf.nn.relu(conv2d(x_obs, W_conv1) + b_conv1) #size = [None, 1, 2, 5]

#fully connected layer
W_fc1 = weight_variable([1*2*5, 16]) 
b_fc1 = bias_variable([16])

#flatten previous layer's output to apply fc layer
h_conv1_flat = tf.reshape(h_conv1, [-1, 1*2*5]) #size = [None, 10]

#apply fc layer
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1) #size = [None, 16]

# (no dropout)
#keep_prob = tf.placeholder(tf.float32) #keep probability taken from feed
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#final readout 
W_fc2 = weight_variable([16, 3])
b_fc2 = bias_variable([3])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2 #size = [None, 3]

opt_act = tf.argmax(y_conv,1)
maxq = tf.reduce_max(y_conv)

#total cost (un-normalized)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)

#total cost (normalized)
cross_entropy = tf.reduce_mean(cost)
tf.summary.scalar('cross_entropy', cross_entropy)

#train op (using ADAM optimizer)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

summ_merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("tf/summaries/", sess.graph)

#-----------------------------------------Neural net: Freeze copy----------------------------#

#first convolution layer params
fW_conv1 = weight_variable([1, 1, 1, 5])
fb_conv1 = bias_variable([5])

#apply first convolution (no pooling)
fh_conv1 = tf.nn.relu(conv2d(x_obs, fW_conv1) + fb_conv1) #size = [None, 1, 2, 5]

#fully connected layer
fW_fc1 = weight_variable([1*2*5, 16])
fb_fc1 = bias_variable([16])

#flatten previous layer's output to apply fc layer
fh_conv1_flat = tf.reshape(fh_conv1, [-1, 1*2*5]) #size = [None, 10]

#apply fc layer
fh_fc1 = tf.nn.relu(tf.matmul(fh_conv1_flat, fW_fc1) + fb_fc1) #size = [None, 16]

# (no dropout)
#keep_prob = tf.placeholder(tf.float32) #keep probability taken from feed
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#final readout 
fW_fc2 = weight_variable([16, 3])
fb_fc2 = bias_variable([3])
fy_conv = tf.matmul(fh_fc1, fW_fc2) + fb_fc2 #size = [None, 3]

fopt_act = tf.argmax(fy_conv,1)
fmaxq = tf.reduce_max(fy_conv)

#-----------------------------------------caller functions -----------------------------------#
	
def eps_greedyAction(observation):

	global explore_count	

	[maxq_val, opt_act_val] = sess.run([maxq, opt_act], feed_dict={x:observation})
	opt_act_val = opt_act_val[0]
	if explore_count > explore_limit:
		opt_act_val = randint(0,2)
		explore_count = 0
		print "explore"
	return [maxq_val, opt_act_val]

def allAction(observation):
	y_conv_val = sess.run(y_conv, feed_dict={x:observation})
	return y_conv_val

def fmaxAction(observation):
	[fmaxq_val, fopt_act_val] = sess.run([fmaxq, fopt_act], feed_dict={x:observation})
	fopt_act_val = fopt_act_val[0]
	return [fmaxq_val, fopt_act_val]

def Freeze():
	global fW_conv1, W_conv1, fb_conv1, b_conv1, fW_fc1, W_fc1, fb_fc1, b_fc1, fW_fc2, W_fc2, fb_fc2, b_fc2 
	fW_conv1 = tf.assign(fW_conv1, W_conv1)
	fb_conv1 = tf.assign(fb_conv1, b_conv1)
	fW_fc1 = tf.assign(fW_fc1, W_fc1)
	fb_fc1 = tf.assign(fb_fc1, b_fc1)
	fW_fc2 = tf.assign(fW_fc2, W_fc2)
	fb_fc2 = tf.assign(fb_fc2, b_fc2)
	sess.run([fW_conv1, fb_conv1, fW_fc1, fb_fc1, fW_fc2, fb_fc2])

def Update():
	global tim
	for j in range(max_samples):
		#get random sample from replay memory
		if rm_full_flag == 1:
			sample_index = randint(0,rm_size-1)
		else:
			sample_index = randint(0,rm_reset_count)
		
		#print sample_index
		ob = rm_present_state[sample_index]
		act = rm_action[sample_index] 
		rew = rm_reward[sample_index]
		nob = rm_next_state[sample_index]

		#calculate the max fixed target
		[f_qest,f_act] = fmaxAction(nob)
		tar = rew + df*f_qest

		print ("tar = %f" %tar)		

		#calculate the guess targets - *Doubt here*: should output of cnn be one q value, with (s,a) pairs as input of dim[2,3] ??
		u_qest = allAction(ob)

		#final ylabel	
		u_qest[0][act] = tar	

		#update params
		for i in range(max_iters):
			sess.run(train_step, feed_dict={x:ob, y_:u_qest})	
			#add record to summary
			summ_run = sess.run(summ_merge, feed_dict={x:ob, y_:u_qest})
			writer.add_summary(summ_run,tim)		

#---------------------------------------------------------Main------------------------------------------------------#
def main():
	#init op
	init = tf.global_variables_initializer()

	#run init op
	sess.run(init)
	
	global explore_count
	global rm_reset_count
	global rm_full_flag
	global tim
	freerun_count = -1 #since Freeze() should run the first time 
	for i_episode in range(1000):
		observation = env.reset()

		for t in range(10000):
			env.render()
			tim += 1
			explore_count += 1
			rm_reset_count += 1
			freerun_count += 1

			if rm_reset_count >= rm_size:
				rm_reset_count = 0
				print "replay memory reset"
				rm_full_flag = 1
		
			#get q value estimate and optimal action for present state according to new policy
			[est_q, opt_act] = eps_greedyAction(observation) 

			#make copy of present observation
			ob_copy = observation
  
			# take optimal action
			# get immediate reward and next state's observation
			observation, reward, done, info = env.step(opt_act) 

			#store in replay memory
			rm_present_state[rm_reset_count] = ob_copy
			rm_action[rm_reset_count] = opt_act
			rm_reward[rm_reset_count] = reward
			rm_next_state[rm_reset_count] = observation
				
			if freerun_count % freerun_maxsteps == 0:
				Freeze()
				print "params updated"

			Update()					

			if done:

				break;

	env.close()

if __name__ == '__main__':
	main()

