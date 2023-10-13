from collections import deque
import numpy as np

class PID():
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.1, buffer=10) -> None:
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=buffer)

    @property
    def Kp(self):
        return self._k_p
    
    @Kp.setter
    def Kp(self, value):
        self._k_p = value

    @property
    def Ki(self):
        return self._k_i
    
    @Ki.setter
    def Ki(self, value):
        self._k_i = value

    @property
    def Kd(self):
        return self._k_d
    
    @Kd.setter
    def Kd(self, value):
        self._k_d = value

    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self, value):
        self._dt = value

    def run_step(self, target:float, current:float) -> float:
        error = target - current
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            d_term = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            i_term = sum(self._error_buffer) * self._dt
        else:
            d_term = 0.0
            i_term = 0.0

        return np.clip((self._k_p * error) + (self._k_d * d_term) + (self._k_i * i_term), -1.0, 1.0)