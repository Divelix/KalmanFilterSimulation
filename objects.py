import vrep
import numpy as np


def rotate(angle, desired_angle):
    xa = [np.cos(angle), np.sin(angle)]
    rot = np.array([[np.cos(-desired_angle), -np.sin(-desired_angle)], [np.sin(-desired_angle), np.cos(-desired_angle)]])
    delta_v = np.dot(rot, xa)
    delta = np.arctan2(delta_v[1], delta_v[0])
    return delta


class Robot:
    left_wheel = None
    right_wheel = None
    track_width = 0
    speed = [0, 0]
    angle = 0.0
    coordinates = [0.0, 0.0]

    def __init__(self, left_wheel, right_wheel, track_width=0):
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        self.track_width = track_width

    def get_position(self):
        s_l, d_s_l = self.left_wheel.get_distance()
        s_r, d_s_r = self.right_wheel.get_distance()
        d_s_d = (d_s_l + d_s_r) / 2
        angle = (s_r - s_l) / self.track_width
        delta_angle = rotate(angle, self.angle)
        self.angle += delta_angle
        self.coordinates[0] += np.cos(angle) * d_s_d
        self.coordinates[1] += np.sin(angle) * d_s_d
        return self.coordinates, self.angle

    def get_speed(self):
        return self.speed

    def set_velocity(self, left, right):
        self.left_wheel.set_velocity(left)
        self.right_wheel.set_velocity(right)

    def set_velocity_vec(self, vel):
        def_speed = vel[0]
        side = np.sign(vel[1]) # +left, -right
        left = right = def_speed

        # имитация гироскопа
        real_angle = self.left_wheel.get_true_orientation()
        # print('real: {}; needed: {}'.format(real_angle[2], np.arctan2(vel[1],vel[0])))
        residual = np.arctan2(vel[1],vel[0]) - real_angle[2]
        p = 10 # скорость переходного процесса
        add_speed = np.abs(vel[1]) * np.sign(vel[0]) * np.abs(residual*p)

        if side > 0:
          right += add_speed
        elif side < 0:
          left += add_speed
        self.set_velocity(left, right)

    def get_true_position(self):
        l = self.left_wheel.get_true_position()[:2]
        r = self.right_wheel.get_true_position()[:2]
        return [(l[0]+r[0])/2,(l[1]+r[1])/2]


class Wheel:
    client_id = -1
    handle = 0
    radius = 0
    angle = 0.0
    position = [0,0,0]
    orientation = [0,0,0]

    def __init__(self, client_id, handle, radius=0):
        self.client_id = client_id
        self.handle = handle
        self.radius = radius
        _, self.angle = vrep.simxGetJointPosition(self.client_id, self.handle, vrep.simx_opmode_streaming)
        _, self.position = vrep.simxGetObjectPosition(self.client_id, self.handle, -1, vrep.simx_opmode_streaming)
        _, self.orientation = vrep.simxGetObjectOrientation(self.client_id, self.handle, -1, vrep.simx_opmode_streaming)

    def set_velocity(self, speed):
        vrep.simxSetJointTargetVelocity(self.client_id, self.handle, speed, vrep.simx_opmode_oneshot)

    def get_angle(self):
        _, angle = vrep.simxGetJointPosition(self.client_id, self.handle, vrep.simx_opmode_buffer)
        delta_angle = rotate(angle, self.angle)
        self.angle += delta_angle
        return self.angle, delta_angle

    def get_distance(self):
        angle, delta_angle = self.get_angle()
        distance = angle * self.radius
        delta_distance = delta_angle * self.radius
        return distance, delta_distance

    def get_true_position(self):
        _, self.position = vrep.simxGetObjectPosition(self.client_id, self.handle, -1, vrep.simx_opmode_buffer)
        return self.position

    def get_true_orientation(self):
        _, self.orientation = vrep.simxGetObjectOrientation(self.client_id, self.handle, -1, vrep.simx_opmode_buffer)
        return self.orientation