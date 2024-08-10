import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Load environment with proper wrapping
env = gym.make('ALE/Frogger-v5')
env = Monitor(env)  # Using Monitor from stable_baselines3

# Create the DQN agent with adjusted hyperparameters
model = DQN('CnnPolicy', env, verbose=1, buffer_size=50000, learning_starts=1000, target_update_interval=500, learning_rate=0.0005, batch_size=32)

rewards = []
num_episodes = 10   
timesteps_per_episode = 7000   
for episode in range(num_episodes):
    model.learn(total_timesteps=timesteps_per_episode, log_interval=4)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)  # Reduced evaluation frequency and number of episodes
    rewards.append(mean_reward)
    print(f"Episode {episode+1}/{num_episodes} - Mean reward: {mean_reward} +/- {std_reward}")

# Save the model
model.save("dqn_frogger")

# Plot rewards over time
plt.plot(rewards)
plt.xlabel('Training Episode')
plt.ylabel('Mean Reward')
plt.title('Training Progress')
plt.savefig('training_progress_v3.png')
# plt.show()

# Final evaluation
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
print(f"Final evaluation - Mean reward: {mean_reward} +/- {std_reward}")


env.close()  # Ensure environment is properly closed
