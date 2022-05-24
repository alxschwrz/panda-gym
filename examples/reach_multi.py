import gym
import panda_gym
import time

env = gym.make("PandaReachMulti-v2", render=True)
for _ in range(10):
    obs = env.reset()
    done = False
    print("starting episode")
    while not done:
        action = env.action_space.sample()
        action = obs['desired_goal'] - obs['achieved_goal']
        obs, reward, done, info = env.step(action)
        print(obs['achieved_goal'], obs['desired_goal'], reward)
        env.render()
        time.sleep(0.1)


env.close()
