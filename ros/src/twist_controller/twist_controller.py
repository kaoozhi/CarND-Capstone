from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

import rospy
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass,fuel_capacity, brake_deadband,decel_limit,accel_limit,wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):

        # Initialize yaw controller
        min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # Initialize PID throttle controller
        kp =0.7
        ki =0.1
        kd = 0.
        throttle_min = 0.
        throttle_max = 0.5

        self.throttle_controller = PID(kp, ki, kd, throttle_min, throttle_max)

        # Initialize low pass filter on velocity
        tau = 0.5
        ts = 0.02
        self.lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        # self.steer_ratio = steer_ratio
        # self.max_lat_accel = max_lat_accel
        # self.max_steer_angle = max_steer_angle
        # TODO: Implement
        # pass

        self.prev_time = rospy.get_time()

    def control(self, cur_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # throttle controler
        # low pass filter of velocity
        # steering controler
        # if dbw not enabled deactivate the controller
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        cur_vel = self.lpf.filt(cur_vel)     
        vel_err = linear_vel - cur_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.prev_time
        self.prev_time = current_time

        throttle = self.throttle_controller.step(vel_err, sample_time)

        brake = 0
        if linear_vel ==0 and cur_vel <0.1:
            throttle = 0
            brake = 700 # torque to maintain car stopped
        elif throttle < 0.1 and vel_err < 0:
            throttle = 0
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # torque = F*r = m*a*r

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, cur_vel)

        return throttle, brake, steering
