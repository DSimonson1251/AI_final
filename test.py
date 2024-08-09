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
model = DQN(
    'CnnPolicy', env, verbose=1, 
    buffer_size=50000,      # Reduced buffer size
    learning_starts=5000,   # Increased learning start steps
    target_update_interval=1000, 
    learning_rate=0.0001, 
    batch_size=64,          # Increased batch size
    train_freq=4            # Train every 4 steps
)

# Train the agent and record rewards
rewards = []
num_episodes = 10  # Reduced number of training episodes
timesteps_per_episode = 10000  # Reduced timesteps per episode

for episode in range(num_episodes):
    model.learn(total_timesteps=timesteps_per_episode, log_interval=4)
    
    # Evaluate only every 2 episodes
    if episode % 2 == 0:
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
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"Final evaluation - Mean reward: {mean_reward} +/- {std_reward}")

# No video recording during training
env.close()  # Ensure environment is properly closed
