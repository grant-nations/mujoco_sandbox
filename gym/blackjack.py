from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class BlackJackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        """
        Choose an action from an observation of (player_sum, dealer_card, usable_ace).

        Returns 0 for stand, 1 for hit
        """

        # With probability epsilon: explore (choose random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (choose percieved best action)
        else:
            return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_observation: tuple[int, int, bool],
    ):

        # What's the best we could to from the next state? (0 if terminated)
        future_q_value = (not terminated) * np.max(self.q_values[next_observation])

        # What should the q value be based on the actual reward we got?
        target_q_value = reward + self.discount_factor * future_q_value

        # How wrong was the current estimate?
        q_value_error = target_q_value - self.q_values[observation][action]

        # Update our estimate in the direction of the error
        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.lr * q_value_error
        )

        self.training_error.append(q_value_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Test the trained agent
def test_agent(agent: BlackJackAgent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 100_000
    start_epsilon = 1.0  # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1  # Always keep some exploration

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

    agent = BlackJackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        observation, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.update(
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                next_observation=next_observation,
            )

            done = terminated or truncated
            observation = next_observation

        agent.decay_epsilon()




    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

    test_agent(agent, env)