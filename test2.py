import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Definir la red neuronal para el agente DQL
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Función epsilon-greedy para seleccionar acciones
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        with torch.no_grad():
            q_values = policy_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

# Parámetros
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
gamma = 0.99
learning_rate = 0.001
num_episodes = 5000
max_steps_per_episode = 500
batch_size = 32
target_update_freq = 100

# Crear el entorno
env = gym.make('MountainCar-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Inicializar la red neuronal del agente y el optimizador
policy_net = DQN(input_size, output_size)
target_net = DQN(input_size, output_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Bucle principal de entrenamiento
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        # Seleccionar acción
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / 200)
        action = select_action(state, epsilon)
        
        # Aplicar acción y obtener siguiente estado, recompensa y si el episodio ha terminado
        next_state, reward, done, _ = env.step(action)
        
        # Actualizar la red neuronal
        optimizer.zero_grad()
        q_values = policy_net(torch.tensor(state, dtype=torch.float32))
        q_value = q_values[action]
        with torch.no_grad():
            next_q_value = torch.max(target_net(torch.tensor(next_state, dtype=torch.float32)))
            target = reward + gamma * next_q_value * (1 - done)
        loss = nn.MSELoss()(q_value, target)
        loss.backward()
        optimizer.step()
        
        total_reward += reward
        state = next_state
        
        # Actualizar el target network
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break
    
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

# Simulación de la política aprendida
state = env.reset()
done = False

while not done:
    env.render()
    action = select_action(state, 0.01)  # Seleccionar acción con epsilon = 0.01 (greedy)
    state, _, done, _ = env.step(action)
    time.sleep(0.05)

# Cerrar el entorno después de la finalización del entrenamiento
env.close()
