from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))


max_t=1000
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]
score = 0

for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]
        agent.step(state,action,reward,next_state,done)
        score += reward                                # update the score
        state = next_state

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

env.close()
