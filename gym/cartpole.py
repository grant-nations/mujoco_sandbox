import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

# Restart environment to start a new episode
observation, info = env.reset()

print(f"Initial observation: {observation}")

total_reward = 0

terminated, truncated = False, False

while not (terminated or truncated):
    # Choose action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Choose a random action

    # Take the action, see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward

print(f"Episode finished; total reward: {total_reward}")
print(f"Terminate: {terminated}")
print(f"Truncated: {truncated}")

env.close()
