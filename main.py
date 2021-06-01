from objects import *
import vrepConst
import sys
import time
import matplotlib.pyplot as plt

vrep.simxFinish(-1)
client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if client_id != -1:
    print('Connected to remote API server')
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
else:
    sys.exit('Failed connecting to remote API server')

# vrep.simxSynchronous(client_id, True) # call once
# vrep.simxSynchronousTrigger(client_id) # each step
# --------------------------------------------------------------
wheel_radius = 0.125
track_width = 0.4

err, left_motor_handle = vrep.simxGetObjectHandle(client_id, 'MotorBL', vrep.simx_opmode_oneshot_wait)
err, right_motor_handle = vrep.simxGetObjectHandle(client_id, 'MotorBR', vrep.simx_opmode_oneshot_wait)

left_wheel = Wheel(client_id, left_motor_handle, wheel_radius)
right_wheel = Wheel(client_id, right_motor_handle, wheel_radius)
robot = Robot(left_wheel, right_wheel, track_width)

dt = 0.1
# Робот двигается на плоскости (x,y)
A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]) # матрица состояния в (x,y)
B = np.array([[dt,0],[0,dt],[1,0],[0,1]]) # матрица управления (по скорости) в (x,y)
x = np.zeros([4,1]) # вектор состояния
dp = 0.01 # начальная ошибка положения (одинаковые для x и y)
dv = 0.01 # начальная ошибка скорости (одинаковые для x и y)
P = np.array([[dp**2,0,dp*dv,0],[0,dp**2,0,dp*dv],[dp*dv,0,dv**2,0],[0,dp*dv,0,dv**2]]) # матрица ковариации
H = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]) # матрица преобразования (т.к. датчик не измеряет скорость)
u = np.zeros([2,1]) # вектор управления
e = 0.001
q = np.array([e,e,e,e])
Q = np.diag(q) # ковариация модели
w = np.random.normal(0,q).reshape(4,1) # ошибка в моделе движения
op = e # ошибка измерения положения (одинаковые для x и y)
R = np.array([[op**2,0,0,0],[0,op**2,0,0],[0,0,0,0],[0,0,0,0]]) # ковариация модели измерения
z = np.random.normal(0,[op,op,0,0]).reshape(4,1) # ошибка датчика
C = np.array([[1,0],[0,1],[0,0],[0,0]])
I = np.eye(4)

v = np.array([[1,0],[1,0.1],[1,-0.1],[1,0.1],[1,-0.2],[0,0]]) # значения для управляющего сигнала
v_i = 0
p_true = np.zeros([50,2])
p_estm = np.zeros([50,2])
p_obs = np.zeros([50,2])
p_calm = np.zeros([50,2])
k_trace = np.zeros([50,1])

for i in range(50):
	# раз в секунду меняем направление движения (управляющее воздействие)
	if i % 10 == 0: 
		v_i += 1
		u = v[v_i].reshape(2,1)
	robot.set_velocity_vec(v[v_i])

	x = A@x + B@u + w
	P = A@P@A.T + Q

	p_estm[i,:] = x[:2].T

	# K = P@H.T / (H@P@H.T + R)
	K = P / (P + R)
	K[np.isnan(K)] = 0
	k_trace[i] = np.trace(K)

	pos = robot.get_position()[0] + np.random.normal(0,e,1) # зашумляем значения с датчика
	y = np.asarray(pos).reshape(2,1)
	z = np.random.normal(0,[op,op,0,0]).reshape(4,1)
	y = C@y + z

	x = x + K@(y - H@x)

	# P = (I - K@H)@P
	P = (I - K)@P

	p_true[i,:] = robot.get_true_position()
	p_obs[i,:] = np.asarray(robot.get_position()[0]) + z[:2].reshape(-1,2)
	p_calm[i,:] = x[:2].T

	time.sleep(dt)

vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)

# plt.plot(k_trace)
fig, ax = plt.subplots()
ax.plot(p_true[:,0], p_true[:,1], 'k', label='Ground truth')
ax.plot(p_estm[:,0], p_estm[:,1], 'b--', label='Motion model')
ax.plot(p_obs[:,0], p_obs[:,1], 'g--', label='Encoder data')
ax.plot(p_calm[:,0], p_calm[:,1], 'r--', label='Kalman filter')
legend = ax.legend(loc='lower left')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# print(p_true)

# x = robot.get_position()[0][0]
# dist = laser_sensor.get_distance()
# print('x = {}, dist  = {}'.format(x, 7-dist))