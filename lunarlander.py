import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import matplotlib.pyplot as plt


def dqn (alpha, input_layer, output_layer, hidden_layer_size):
    '''
    Constructs and returns a Deep Neural Network with 2 hidden layers. This will be used to estimate the Q(s,a).
    Network is designed to recieve a state and estimate Q values assocaited with all available actions at the same time
    Parameters
    ----------
        alpha: float
            learning rate used by the DNN
        input_layer: int
            Size of state space, determines size of first layer
        output_layer: int
            Size of action space, deteremines size of final layer
        hidden_layer_size: int
            Number of nodes in the hidden layers
    Returns
    -------
        Keras model
            Deep Neural Network with 4 fully connected layers
    '''
    model = keras.Sequential([keras.layers.Dense(hidden_layer_size, activation='relu', input_shape =(input_layer,)),
                              keras.layers.Dense(hidden_layer_size, activation='relu'),
                              keras.layers.Dense(output_layer,
                                                 activation=None)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= alpha),
                  loss="mean_squared_error")
    return model

class ReplayMemory():
    def __init__(self, length, state_space_size):
        '''
        Constructs a replay memory that will be used to store and sample experiences which will be used to train the
        DNN. This could be approached in various ways, our approach is to construct a separate numpy array for each
        component of the experience tuple.

        Parameters
        ----------
            length: int
                size of replay memory
            state_space_size: int
                number of variables in the state vector
        '''

        self.length = length
        self.counter = 0
        self.state_memory = np.zeros((length, state_space_size), dtype=np.float32)
        self.prime_memory = np.zeros((length, state_space_size), dtype=np.float32)
        self.actions_memory = np.zeros(self.length, dtype=np.int32)
        self.rewards_memory = np.zeros(self.length, dtype=np.float32)
        self.terminal_memory = np.zeros(self.length, dtype=np.int32)

    def store(self, state, action, reward, s_prime, done):
        '''
        Method for updating replay memory. Adds new experience tuple. If memory is already full it replaces the oldest
        experience

        Parameters
        ----------
            state: array (float), shape (state_space_size,)
                observed state.
            action: int,
                action taken.
            reward: float
                observed reward for taking action a in state s
            s_prime: array (float), shape (state_space_size,)
                observed new state when taking action a in state s
            done: Boolean
                whether episode terminates when taking action a in state s

        '''

        ## create index of experience in replay memory. If counter > length, replace oldest expereince
        index = self.counter % self.length
        ## Save the <s,a,r,s'> tuple as well as terminal boolean each in its respective array
        self.state_memory[index] = state
        self.actions_memory[index] = action
        self.prime_memory[index] = s_prime
        self.rewards_memory[index] = reward
        self.terminal_memory = done
        ## increment count
        self.counter +=1

    def sample(self, batch_size):
        '''
        Method for randmoly sampling a batch of experiences from replay memory. This batch is then used to update the
        DNN weights.

        Parameters
        ----------
            batch_size: int
                the size of the batch to be randomly sampled

        Returns
        -------
            state_batch: array (float), shape (batch_size, state_space_size)
                states of the randomly sampled batch of experiences
            action_batch: array (int), shape(batch_size, 1)
                actions of the randomly sampled batch of experiences
            rewards_batch: array (float), shape(batch_size, 1)
                rewards of the randomly sampled batch of experiences
            prime_batch: array (float), shape (batch_size, state_space_size)
                transitional states of the randomly sampled batch of experiences
            terminal_batch: array (Boolean), shape(batch_size, 1)
                termination of the randomly sampled batch of experiences
        '''
        ## If memory is not yet full we need to sample from 0 to counter, otherwise from entire memory
        limit = min(self.counter, self.length)
        ## Randomly sample indexes of size batch_size
        samples = np.random.choice(limit, batch_size, replace=False)
        ## Return experiences using those indexes
        return self.state_memory[samples], self.actions_memory[samples], self.rewards_memory[samples],\
               self.prime_memory[samples], self.terminal_memory[samples]

class Agent():
    def __init__(self,state_space_size, action_space_size, learning_rate, discount_rate, random_action_rate,
                 random_action_decay_rate, min_random_action, replay_memory_size, batch_size, hidden_layer_size):
        '''
        Constructs an DQN agent with a replay memory and a Deep Neural Network
        Parameters
        ----------
            state_space_size: int
                length of state vector

            action_space_size: int
                Number of available actions
            learning_rate: float
                alpha parameter used to initalize Deep Neural Network
            discount_rate: float
                gamma used to discount expected reward of transitional states
            random_action_rate: float
                initial rate of exploration
            random_action_decay_rate: float
                rate of exploration decay, should be less than 1
            min_random_action: float
                minimum exploration rate
            replay_memory_size: int
                size of replay memory, used for initalizing replay memory
            batch_size: int
                batch size to be used when sampling from replay memory
            hidden_layer_size: int
                number of nodes per hidden layer, used to initialize neural network
        '''
        self.actions = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_rate
        self.rar = random_action_rate
        self.radr = random_action_decay_rate
        self.min_rar = min_random_action
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(replay_memory_size, state_space_size )
        self.DQN = dqn(learning_rate, state_space_size, action_space_size, hidden_layer_size)

    def act(self, state):
        '''
        method for choosing the action using an epsilon-greedy policy

        Parameters
        ----------
            state: array (float), shape (state_space_size,)
                agent's current state
        Returns
        -------
            action: int
                action to be taken by agent in current state
        '''

        ## explore with probability rar
        if np.random.random() < self.rar:
            ## choose a legal action at random
            action = np.random.randint(0, self.actions)

            ## decay exploration at rate radr up to min_rar
            self.rar = self.rar * self.radr if self.rar > self.min_rar else self.min_rar
        ## otherwise greedily exploit
        else:
            ## reshape state so that it 1x8 which is what DQN expects for input
            state = np.array([state])
            ## Get the approximated Q values for all actions from DQN
            actions = self.DQN.predict(state)
            ## Choose the action with the maximum value out of all possible actions
            action = np.argmax(actions)
        return action

    def memorize(self, state, action, reward, s_prime, done):
        '''
        Interface method between agent and replay memory for stroing experiences
        '''
        self.replay_memory.store(state, action, reward, s_prime, done)

    def learn(self):
        '''
        Method for updating the weights of the DNN

        '''
        ## make sure memory has filled up to at least batch size
        if self.replay_memory.counter >= self.batch_size:
            ## randomly sample a batch size from replay_memory so we can train the learner
            states, actions, rewards, s_primes, dones = self.replay_memory.sample(self.batch_size)
            ## states to be updated
            Q_states = self.DQN.predict(states)
            ## get approximated Q values of s primes to bootstrap future rewards
            Q_s_primes = self.DQN.predict(s_primes)
            ## make a copy of states which will be passed to DQN for weight updates
            Q_targets = np.copy(Q_states)
            index = np.arange(self.batch_size)
            ##assign new rewards to the states
            ##if state is terminal dones = 0, no future rewards
            Q_targets[index, actions] = rewards + self.gamma *np.max(Q_s_primes, axis=1)*(1-int(dones))
            self.DQN.train(Q_states, Q_targets)


def plot_scatter(x, y, title, xlabel, ylabel, filename):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.xlabel(xlabel, fontweight = 'bold')
    plt.ylabel(ylabel, fontweight = 'bold')
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

if __name__ =='__main__':
    ## disable tensorflow's eager execution to expedite the NN's learning
    tf.compat.v1.disable_eager_execution()

    ## initiate lunar lander environment
    env = gym.make('LunarLander-v2')

    ## agent parameters
    alpha = 0.01
    gamma =0.9
    rar = 0.5
    radr = 0.99
    min_rar = 0.01
    hidden_layer_size = 128

    ## training episodes
    episodes = 1000

    ## array for saving score of each episodes
    scores = np.empty(1000)

    ## initialize DQN agent
    agent = Agent(state_space_size=env.observation_space.n, action_space_size=env.action_space.n, learning_rate=alpha,
                  discount_rate=gamma, random_action_rate=rar, random_action_decay_rate=radr, min_random_action=min_rar,
                  replay_memory_size=1000000, batch_size=64, hidden_layer_size=hidden_layer_size)

    ## train agent
    for i in range (episodes):
        done = False
        total_reward = 0
        state = env.reset()
        while not done:
            action = agent.act(state)
            s_prime, reward, done, info = env.step(action)
            total_reward +=reward
            agent.memorize(state, action, reward, s_prime, done)
            state = s_prime
            agent.learn()
        scores[i] = total_reward
    ## plot scores
    plot_scatter(np.arange(1, episodes+1), scores, "Training Episodes' Score", "episode", "score",
                 "training scores.png")


