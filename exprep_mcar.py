import gym
from random import randint
env = gym.make('MountainCar-v0')

#initial weights (dimension according to two state variables and 3 discrete actions)
#new weights
nw1 = [0, 0, 0]
nw2 = [0, 0, 0]
nw3 = [0, 0, 0]
#old weights
ow1 = [0, 0, 0]
ow2 = [0, 0, 0]
ow3 = [0, 0, 0]
 
alpha = 0.75 #learning rate
df = 0.99 #discount factor 
explore_count = 0 
explore_limit = 100
freerun_maxsteps = 50
rm_reset_count = -1
max_samples = 60
max_iters = 1
rm_size = 100000
rm_full_flag = 0

#replay memory
rm_present_state = [[0,0]]*rm_size
rm_action = [0]*rm_size
rm_reward = [0]*rm_size
rm_next_state = [[0,0]]*rm_size

def calq(ob, act):
	if act == 0:
		w = nw1
	elif act == 1:
		w = nw2
	else:
		w = nw3
	q = w[0]+ob[0]*w[1]+ob[1]*w[2]
	return q

def ocalq(ob, act):
	if act == 0:
		w = ow1
	elif act == 1:
		w = ow2
	else:
		w = ow3
	q = w[0]+ob[0]*w[1]+ob[1]*w[2]
	return q
	
def eps_greedyAction(observation):
	global explore_count	
	if explore_count > explore_limit:
		opt_act = randint(0,2)
		maxq = calq(observation,opt_act)
		explore_count = 0
		print "explore"
		return [maxq, opt_act]

	q1 = calq(observation,0)
	q2 = calq(observation,1)
	q3 = calq(observation,2)
	maxq = 0
	opt_act = 3

	if q2>q1:
		if q2>q3:
			maxq = q2
			opt_act = 1
		else:
			maxq = q3
			opt_act = 2
	else:
		if q1>q3:

			maxq = q1
			opt_act = 0
		else:
			maxq = q3
			opt_act = 2

	return [maxq, opt_act]

def maxAction(observation):
	q1 = ocalq(observation,0)
	q2 = ocalq(observation,1)
	q3 = ocalq(observation,2)
	maxq = 0
	opt_act = 3
	if q2>q1:
		if q2>q3:
			maxq = q2
			opt_act = 1
		else:
			maxq = q3
			opt_act = 2
	else:
		if q1>q3:

			maxq = q1
			opt_act = 0
		else:
			maxq = q3
			opt_act = 2
	return [maxq, opt_act]


def Update1(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	nw1[0] -= tmp*1
	nw1[1] -= tmp*ob[0]
	nw1[2] -= tmp*ob[1]

def Update2(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	nw2[0] -= tmp*1
	nw2[1] -= tmp*ob[0]
	nw2[2] -= tmp*ob[1]

def Update3(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	nw3[0] -= tmp*1
	nw3[1] -= tmp*ob[0]
	nw3[2] -= tmp*ob[1]

def Update():

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

		#calculate fixed target
		[f_qest,f_act] = maxAction(nob)
		tar = rew + df*f_qest

		#update params
		for i in range(max_iters):
		#	[qest, opt_act] = eps_greedyAction(ob) 	
		#	if opt_act == 0:
		#		Update1(qest,tar,ob)
		#	elif opt_act ==1: 
		#		Update2(qest,tar,ob)
		#	else:
		#		Update3(qest,tar,ob)	
			if act == 0:
				qest = nw1[0]+ob[0]*nw1[1]+ob[1]*nw1[2] #ok though redundant
				Update1(qest,tar,ob)
			elif act ==1: 
				qest = nw2[0]+ob[0]*nw2[1]+ob[1]*nw2[2]
				Update2(qest,tar,ob)
			else:
				qest = nw3[0]+ob[0]*nw3[1]+ob[1]*nw3[2]
				Update3(qest,tar,ob)			


def main():

	global explore_count
	global rm_reset_count
	global rm_full_flag
	freerun_count = 0
	for i_episode in range(1000):
		observation = env.reset()
		
		for t in range(10000):
			env.render()

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
				#freeze params
				ow1 = nw1
				ow2 = nw2
				ow3 = nw3
				print "params updated"

			Update()					

			if done:

				break;

	env.close()

if __name__ == '__main__':
	main()

