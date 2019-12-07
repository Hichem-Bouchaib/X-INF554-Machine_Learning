import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Meta parameters for the RL agent
alpha = 0.1
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.999
verbose = True

# Define types of algorithms
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
EPSILON_GREEDY = "epsilon_greedy"
SOFTMAX = "softmax"
GREEDY = "greedy"

# Choose methods for learning and exploration
rl_algorithm = SARSA #Q_LEARNING
explore_method = EPSILON_GREEDY

# Draw a softmax sample
def softmax(q):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

# Act with softmax
def act_with_softmax(s, q):
    prob_a = softmax(q[s, :])
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q):
    a = np.argmax(q[s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q.shape[1])
    return a

# Compute SARSA update
def sarsa_update(q,s,a,r,s_prime,a_prime):
    td = r + gamma * q[s_prime,a_prime] - q[s,a]
    return q[s,a] + alpha * td

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime):
    td = r + gamma * np.max(q[s_prime, :]) - q[s, a]
    return q[s,a] + alpha * td

# Evaluate a policy on n runs
def evaluate_policy(q,env,n,h,explore_type):
    success_rate = 0.0
    mean_return = 0.0

    for i in range(n):
        discounted_return = 0.0
        s = env.reset()

        for step in range(h):
            if explore_type == GREEDY:
                s,r, done, info = env.step(np.argmax(q[s,:]))
            elif explore_type == EPSILON_GREEDY:
                s,r, done, info = env.step(act_with_epsilon_greedy(s,q))
            elif explore_type == SOFTMAX:
                s,r, done, info = env.step(act_with_softmax(s,q))
            else:
                raise ValueError("Wrong Explore Method in evaluation:".format(explore_type))

            discounted_return += np.power(gamma,step) * r

            if done:
                success_rate += float(r)/n
                mean_return += float(discounted_return)/n
                break

    return success_rate, mean_return

def main():

    global epsilon
    global tau

    #Choose environment
    env_name = 'FrozenLake-v0'

    # Random seed
    np.random.RandomState(42)

    # Create Environment
    env = gym.make(env_name)

    # Recover State-Action space size
    n_a = env.action_space.n
    n_s = env.observation_space.n

    # Experimental setup
    n_episode = 10000
    print("n_episode ", n_episode)
    max_horizon = 100
    eval_steps = 10

    # Monitoring perfomance
    window = deque(maxlen=100)
    last_100 = 0

    greedy_success_rate_monitor = np.zeros([n_episode,1])
    greedy_discounted_return_monitor = np.zeros([n_episode,1])

    behaviour_success_rate_monitor = np.zeros([n_episode,1])
    behaviour_discounted_return_monitor = np.zeros([n_episode,1])

    # Init Q-table
    q_table = np.zeros([n_s, n_a])

    env.reset()

    # Train for n_episode
    for i_episode in range(n_episode):

        # Reset a cumulative reward for this episode
        total_return = 0.0

        # Start a new episode and sample the initial state
        s = env.reset()

        # Select the first action in this episode
        if explore_method == SOFTMAX:
            a = act_with_softmax(s, q_table)
        elif explore_method == EPSILON_GREEDY:
            a = act_with_epsilon_greedy(s, q_table)
        else:
            raise ValueError("Wrong Explore Method:".format(explore_method))


        for i_step in range(max_horizon):

            # Act
            s_prime, r, done, info = env.step(a)

            total_return += np.power(gamma,i_step) *r

            # Select an action
            if explore_method == SOFTMAX:
                a_prime = act_with_softmax(s_prime, q_table)
            elif explore_method == EPSILON_GREEDY:
                a_prime = act_with_epsilon_greedy(s_prime, q_table)
            else:
                raise ValueError("Wrong Explore Method:".format(explore_method))

            # Update a Q value table
            if rl_algorithm == SARSA:
                q_table[s, a] = sarsa_update(q_table,s,a,r,s_prime,a_prime)
            elif rl_algorithm == Q_LEARNING:
                q_table[s, a] = q_learning_update(q_table,s,a,r,s_prime)
            else:
                raise ValueError("Wrong RL algorithm:".format(rl_algorithm))

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                window.append(r)
                last_100 = window.count(1)

                greedy_success_rate_monitor[i_episode-1,0], greedy_discounted_return_monitor[i_episode-1,0]= evaluate_policy(q_table,env,eval_steps,max_horizon,GREEDY)
                behaviour_success_rate_monitor[i_episode-1,0], behaviour_discounted_return_monitor[i_episode-1,0] = evaluate_policy(q_table,env,eval_steps,max_horizon,explore_method)
                if verbose:
                    print("Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}\tEpsilon: {4:.3f}\tSuccess Rate: {5:.3f}\tLast_100: {6}".format(i_episode, i_step, total_return, r, epsilon,greedy_success_rate_monitor[i_episode-1,0],last_100))
                    #print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tTermR: {3}\ttau: {4:.3f}".format(i_episode, i_step, total_return, r, tau)

                break


        # Schedule for epsilon
        epsilon = epsilon * epsilon_decay
        # Schedule for tau
        tau = init_tau + i_episode * tau_inc

    plt.figure(0)
    plt.plot(range(0,n_episode,10),greedy_success_rate_monitor[0::10,0])
    plt.title("Greedy policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")

    plt.figure(1)
    plt.plot(range(0,n_episode,10),behaviour_success_rate_monitor[0::10,0])
    plt.title("Behaviour policy with {0} and {1}".format(rl_algorithm,explore_method))
    plt.xlabel("Steps")
    plt.ylabel("Success Rate")
    plt.show()


    #Show an episod

    # for i_step in range(max_horizon):
    #     env.render()
    #     a = np.argmax(q_table[s, :])
    #     s, r, done, info = env.step(a)
    #     total_return += np.power(gamma,i_step) *r
    #
    #     if done:
    #         print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tFinal_Reward: {3}".format(1, i_step, total_return, r)
    #         break
    #
    # # Show Policy
    #
    # for s in range(n_s):
    #     actions = ['LEFT','DOWN','RIGHT','UP']
    #     print(actions[np.argmax(q_table[s, :])])

if __name__ == "__main__":

    main()
