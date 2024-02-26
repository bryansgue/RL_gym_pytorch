import gym
import time

# Crear el entorno CartPole-v1
env = gym.make('MountainCar-v0')

# Reiniciar el entorno
observation = env.reset()

# Bucle principal para interactuar con el entorno de forma continua
while True:
    # Renderizar la animación del entorno
    env.render()

    # Imprimir el número de observación
    print("Observation:", observation)

    # Tomar una acción aleatoria (0: izquierda, 1: derecha)
    action = env.action_space.sample()

    # Aplicar la acción y obtener la observación, recompensa, estado del episodio y otra información
    observation, reward, done, info = env.step(action)

    # Si el episodio ha terminado, reiniciar el entorno
    if done:
        observation = env.reset()

    # Agregar un pequeño retraso para que la animación sea más visible
    time.sleep(0.1)  # Cambia el valor de 0.1 según lo desees

# Nunca llegamos aquí, pero cerramos el entorno para liberar recursos
env.close()
