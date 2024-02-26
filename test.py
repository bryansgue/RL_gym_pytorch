import gym
import numpy as np
import time

# Crear el entorno MountainCar-v0
env = gym.make('MountainCar-v0')

# Definir parámetros para el algoritmo de Q-Learning
num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1
q_table = np.random.uniform(low=-2, high=0, size=(num_states[0], num_states[1], env.action_space.n))
learning_rate = 0.1
discount_rate = 0.99
num_episodes = 5000
max_steps_per_episode = 1000

# Función para discretizar el estado
def discretize_state(state):
    discretized_state = (state - env.observation_space.low) * np.array([10, 100])
    discretized_state = np.round(discretized_state, 0).astype(int)
    return tuple(discretized_state)

# Bucle principal de entrenamiento
for episode in range(num_episodes):
    # Reiniciar el entorno
    state = env.reset()
    done = False

    # Discretizar el estado inicial
    discretized_state = discretize_state(state)

    for step in range(max_steps_per_episode):
        # Tomar una acción basada en el valor máximo de Q para el estado actual
        action = np.argmax(q_table[discretized_state])

        # Aplicar la acción y obtener la observación, recompensa y si el episodio ha terminado
        new_state, reward, done, _ = env.step(action)

        # Discretizar el nuevo estado
        new_discretized_state = discretize_state(new_state)

        # Actualizar el valor de Q para el estado actual y la acción tomada
        if not done:
            max_future_q = np.max(q_table[new_discretized_state])
            current_q = q_table[discretized_state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)
            q_table[discretized_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discretized_state + (action,)] = 0

        # Actualizar el estado actual
        discretized_state = new_discretized_state

        # Imprimir información relevante
        print(f"Episodio: {episode + 1}, Paso: {step + 1}")

        # Si el episodio ha terminado, salir del bucle
        if done:
            break


# Simulación de la política aprendida
state = env.reset()
done = False

while not done:
    env.render()
    discretized_state = discretize_state(state)
    action = np.argmax(q_table[discretized_state])
    state, _, done, _ = env.step(action)
    time.sleep(0.05)

# Cerrar el entorno después de la finalización del entrenamiento
env.close()
