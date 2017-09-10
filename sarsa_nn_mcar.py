import gym
from random import randint
import tensorflow as tf
import numpy as np

env = gym.make('MountainCar-v0')
eps = 0.1 #explore count
alpha = 0.1 #learning rate
df = 0.99 #discount factor

#get interactive session
sess = tf.InteractiveSession()

#initial weights (dimension according to two state variables and 3 discrete actions)
w = tf.Variable(tf.truncated_normal(shape=[2,3], mean=0.0, stddev=0.1), name='w')
b = tf.Variable(0.0, name="b")


#placeholder for correct labels
obs = tf.placeholder(tf.float32, shape=[2,])

#reshape observation feed
obsr = tf.reshape(obs, [1,2])

#estimated q value of present state
q_est = tf.matmul(obsr,w) + b

#optimal action
act_opt = tf.argmax(q_est, dimension=1)
#act_opt = act_opt[0,0]

#target
tar = tf.placeholder(tf.float32, shape=[1,3])

# loss
loss = tf.reduce_sum(tf.square(q_est - tar)) # sum of the squares
tf.summary.scalar('loss', loss)

#train op
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

summ_merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("tf/summaries/", sess.graph)

#init op
init = tf.global_variables_initializer()

#get session
sess = tf.Session()

#run init op
sess.run(init)
	

def main():
	t = 0
	for i_episode in range(100000):
		observation = env.reset()
	
		for t in range(300):
			global eps 
			env.render()

			#choose eps_greedy action according to present policy
			[opt_act, est_q] = sess.run([act_opt,q_est], feed_dict={obs:observation})
			if np.random.rand(1) < eps:
				opt_act[0] = env.action_space.sample()

			print "opt_act : %d"%(opt_act[0])

			#make copy of present observation
			ob_copy = observation
  
			# take optimal action
			# get immediate reward and next state's observation
			observation, reward, done, info = env.step(opt_act[0]) 

			#next state q value estimate
			n_est_q = sess.run(q_est, feed_dict={obs:observation})

			#calculate targetQ
			max_n_est_q = np.max(n_est_q)
			targetQ = est_q
			targetQ[0,opt_act[0]] = reward + df*max_n_est_q

			#update policy
			sess.run(train_step, feed_dict={obs:ob_copy, tar:targetQ})

			#add record to summary
			summ_run = sess.run(summ_merge, feed_dict={obs:ob_copy, tar:targetQ})
			writer.add_summary(summ_run,i_episode*200+t)


			if done:

				print "done time = %d"%(t)

				#reduce eps exploration
				eps = 1./((i_episode/50) + 10)

				#break;
				continue;

	env.close()

if __name__ == '__main__':
	main()

