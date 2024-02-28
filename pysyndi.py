import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from scipy.io import loadmat

# Cargar el archivo .mat
u_ref= loadmat('u_ref_3.mat')
data = loadmat('states_3.mat')
time = loadmat('t_3.mat')

# Acceder a la variable 'states'
states = data['states']

t = time['t']

u = u_ref['u_ref']

# Asignar las variables según los índices especificados
#v = states[0:3, 0:100]
#v = states[3:6, 0:100]
#v = states[6:9, 0:100]
#euler_p = states[9:12, :]
#v = states[12:15, 0:10]
quat = states[15:19, :]
v = states[19:22, 0:250]


v = v.T

dt = 1/30

t_vector = t[0,0:250]

u_train = u[:,0:250]

u_train = u_train.T
# Fit the model

poly_order = 5
threshold = 0.9

library1 = ps.PolynomialLibrary(degree=poly_order)
library2 = ps.FourierLibrary(n_frequencies=2)

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold, fit_intercept=True),
    feature_library=library1,
)
model.fit(v, u=u_train, t=dt)
model.print()






# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(v, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(v, t=dt)

# Plot original data and model prediction
plt.figure(figsize=(10, 6))

# Plot original data
plt.plot(range(len(x_dot_test_computed[:, 1])), x_dot_test_computed[:, 1], label='Original Data', color='blue')

# Plot model prediction
plt.plot(range(len(x_dot_test_predicted[:, 1])), x_dot_test_predicted[:, 1].T, label='Model Prediction', linestyle='--', color='red')

plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Comparison of Original Data and Model Prediction')
plt.legend()
plt.grid(True)
plt.show()

