import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def initialize_q(pos_space, vel_space, ang_space, ang_vel_space, action_space):
    return np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, action_space.n))

def train_q_learning(env, q, pos_space, vel_space, ang_space, ang_vel_space, is_training=True, learning_rate=0.1, discount_factor=0.99, epsilon=1, epsilon_decay_rate=0.00001):
    rewards_per_episode = []
    rng = np.random.default_rng()
    i = 0

    while True:
        state = env.reset()[0]      
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        rewards = 0
        terminated = False          

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] += learning_rate * (
                    reward + discount_factor*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av

            rewards += reward

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if i % 100 == 0:
            print(f'Episode: {i} Rewards: {rewards} Epsilon: {epsilon:0.2f} Mean Rewards: {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    return q, rewards_per_episode


def test_q_learning(env, q, pos_space, vel_space, ang_space, ang_vel_space):
    rewards_per_episode = []
    i = 0

    while True:
        state = env.reset()[0]      
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        rewards = 0
        terminated = False          

        while not terminated and rewards < 10000:
            action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            state = new_state
            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av

            rewards += reward

            #env.render()  # Render the environment

        rewards_per_episode.append(rewards)

        #if i % 100 == 0:
        print(f'Test Episode: {i}  Rewards: {rewards}')

        if i >= 1000:  # Adjust the number of episodes as needed
            break

        i += 1

    return rewards_per_episode


def save_q_table(q, filename='cartpolex.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(q, f)

def load_q_table(filename='cartpole.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_mean_rewards(rewards_per_episode, filename='cartpole.png'):
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.plot(mean_rewards)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    #env = gym.make('CartPole-v1')
    env = gym.make('CartPole-v1', render_mode='human') 
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)
    action_space = env.action_space
    
    training = False

    if training == True:
        q = initialize_q(pos_space, vel_space, ang_space, ang_vel_space, action_space)

        q_trained, rewards_train = train_q_learning(env, q, pos_space, vel_space, ang_space, ang_vel_space)

        save_q_table(q_trained)

        plot_mean_rewards(rewards_train)
    else:

        print('INICIA TEST')

        # Load the trained Q-values
        q_trained = load_q_table('cartpole.pkl')

        # Test the trained Q-values
        rewards_test = test_q_learning(env, q_trained, pos_space, vel_space, ang_space, ang_vel_space)
        
        # Print and plot the mean rewards
        mean_rewards_test = np.mean(rewards_test)
        print(f'Mean Rewards (Test): {mean_rewards_test:0.1f}')

        plt.plot(rewards_test)
        plt.title('Test Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.show()

    env.close()
