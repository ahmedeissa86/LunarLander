<h1> <strong> LunarLander </strong> </h1>
<h3> Using DQN to solve Lunar Lander environment from OpenAI gym </h3>
<br>
<h4> <u>The challenge:</u><h4>
<p> The primary challenge presented in the Lunar Lander environment is the infinite number of possible state spaces stemming from the six continuous state variables. Overcoming this challenge requires an effective and efficient way of approximating the state space itself and/or approximating the Q values associated with them. <p>
<br>

<h4> <u>The solution:</u><h4>
<p>We use a DQN, a Q-learner with a replay memory connected to a Deep Neural Network (DNN), to solve the Lunar Lander problem. The DQN, or agent, learns by taking an action in a state and calculating the Q(s,a) based on the immediate reward received from the environment plus the bootstrapped Q(s’,a’). The bootstrapped Q(s’,a’) is calculated using a discount factor, γ. This experience (state, action, reward, transition state, whether episode terminates) is then saved in the replay memory. The replay memory is of size 1,000,000 and, once full, older experiences are overwritten by newer one in the order they were inserted into the replay memory.
  
With every step the agent takes a batch of 64 experiences is provided as training samples to the DNN. These experiences are randomly selected in order to eliminate the correlation that exists between sequential experiences when making the weight updates. The DNN uses the ‘relu’ activation function, the Adam optimizer with learning rate, α, and the mean squared error as the loss function. <p>
  
<h4> <u>The results:</u><h4>
<p> The gradual improving in ur agent's score for the first 200+ episodes and its relatively consistent success in solving the problem afterwards as shown training scores figure demonstrates that our agent is effectively learning the right policy. <p>
  
<h4> <u>The drawbacks:</u><h4>

<p> DNNs in general are not guaranteed to converge to the global optima because backpropagation can get stuck in a local minima when performing gradient descent on the loss function. In fact, in this specific application convergence is not at all guaranteed because reinforcement learning defies some of the underlying assumptions supervised learning models, like DNNs, are built on. <p>
