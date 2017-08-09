import gym
from random import randint
env = gym.make('MountainCar-v0')

#initial weights (dimension according to two state variables and 3 discrete actions)
w1 = [0, 0, 0]
w2 = [0, 0, 0]
w3 = [0, 0, 0]
 
wac = 0 #explore count
alpha = 0.01 #learning rate
df = 1 #discount factor

def calq(ob, act):
	if act == 0:
		w = w1
	elif act == 1:
		w = w2
	else:
		w = w3
	q = w[0]+ob[0]*w[1]+ob[1]*w[2]
	return q
	
def eps_greedyAction(observation):
	global wac	
	if wac > 500:
		opt_act = randint(0,2)
		maxq = calq(observation,opt_act)
		wac = 0
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


def Update1(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	w1[0] -= tmp*1
	w1[1] -= tmp*ob[0]
	w1[2] -= tmp*ob[1]

def Update2(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	w2[0] -= tmp*1
	w2[1] -= tmp*ob[0]
	w2[2] -= tmp*ob[1]

def Update3(est,targ,ob):
	err = est - targ
	tmp = alpha*err
	print tmp
	w3[0] -= tmp*1
	w3[1] -= tmp*ob[0]
	w3[2] -= tmp*ob[1]
	

def main():
	for i_episode in range(100000):
		observation = env.reset()
	
		for t in range(100000):
			global wac 
			wac += 1
			env.render()
			
			#get q value estimate and optimal action for present state according to new policy
			[est_q, opt_act] = eps_greedyAction(observation) 

			#make copy of present observation
			ob_copy = observation
  
			# take optimal action
			# get immediate reward and next state's observation
			observation, reward, done, info = env.step(opt_act) 

			#get q value estimate for next state according to present policy
			[f_est_q, f_opt_act] = maxAction(observation)
			
			#calculate target
			target = reward + df*f_est_q

			#update policy
			if opt_act == 0:
				Update1(est_q, target, ob_copy)
			elif opt_act == 1:
				Update2(est_q, target, ob_copy)
			else :
				Update3(est_q, target, ob_copy)

			if done:

				print "done time = %d opt_act = %d"%(t,opt_act)

				break;

	env.close()

if __name__ == '__main__':
	main()

