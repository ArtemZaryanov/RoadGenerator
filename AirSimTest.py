# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import airsim.utils
import numpy as np
import SplineRoad as SR
import CurveController as Controller
import matplotlib.pyplot as plt
from scipy.misc import derivative
# from SplineRoad import function
def direct_vector(client, object_name: str):
    # Двигаемся на малое расстояние, чтобы полуить текущий вектор направления. Его нужно определить лишь один раз\
    car_controls = airsim.CarControls()
    # car_state = client.getCarState()
    pos0 = client.simGetObjectPose(object_name).position.to_numpy_array()
    car_controls.throttle = 1
    client.setCarControls(car_controls)
    time.sleep(0.5)
    pos1 = client.simGetObjectPose(object_name).position.to_numpy_array()
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    car_controls = None
    return pos1 - pos0


def move_to_point_direct(client, object_name: str):
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    car_controls.steering = 0.1
    client.setCarControls(car_controls)
    time.sleep(2)
    car_controls.throttle = 0
    car_controls.steering = 0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    time.sleep(5)


# TODO  ПЕРЕВЕСТИ SPLINE ROAD в class. Реализовать метод get_data. Это нужно будет в любом случае. + контроллер скорости. Обычный if handbrake sleep(0.01) if ....


def velocity_control(v0, min_velocity, max_velocity):
    if v0>= max_velocity:
        return max_velocity - v0
    if v0>=min_velocity and v0<=max_velocity:
        return 0
    if v0<=min_velocity:
        return  min_velocity - v0


def sign_steering(posAgent,p,function):
    # Определение знака поворота
    # Вектор от точки нормали до позиции автомобиля
    BP = np.array([p[0] - posAgent[0],p[1] - posAgent[1]])
    # Вектор касательной
    tan = derivative(function,p[0])
    C = np.array([tan / np.sqrt(1 + tan ** 2), 1 / np.sqrt(1 + tan ** 2)])
    return np.sign(np.cross(C,BP))


def curve_control():
    pass


# Генерация трассы
def generate_track(return_function = False):
    FR = SR.FunctionRoad()
    FR.generate_data()
    FR.move_data() #Потом убрать
    if return_function:
        return SR.get_function(), None
    else:
        return None

# Вывод графиков
def plot_g(positions):
    # client.simPause(True)
    # Рисуем графики Отдельо создание figure и прочее. Чтобы была просто отправка и отрсиовка данных без
    # именований на ходу(может лучше будет
    # Продолжать с того же времени. Значит нудны две переменныхе. Одна для отчета. Друга 60*i, i++
    # Нарисовать прямые. Коридор допустимых скоростей
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(velocity[:, 0], velocity[:, 1])
    axes[0].plot([0, 60], [min_velocity, min_velocity], c='Red', label='min_velocity')
    axes[0].plot([0, 60], [max_velocity, max_velocity], c='Red', label='max_velocity')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('m/c')
    axes[0].set_title('velocity')
    axes[0].legend()
    axes[1].plot(trottles[:, 0], trottles[:, 1])
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('ye')
    axes[1].set_title('throttle')
    axes[2].plot(steerings[:, 0], steerings[:, 1])
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('ye')
    axes[2].set_title('steerings')
    axes[2].legend()
    plt.show()
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(positions[:, 0], positions[:, 1])
    axes[0].plot(np.linspace(0, 400, 100), SR.get_function()(np.linspace(0, 400, 100)), c='red', label="True")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('AgentPosition')
    axes[1].plot(positions[:, 0], steerings[:, 1])
    axes[1].legend()
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('ye')
    axes[1].set_title('steering')
    axes[1].legend()
    # plt.legend()
    plt.show()

# Возврат в начальное состояние и установка автомобиля
# По направлению нормали к трассе
def reset_environment(client,RoadFunction=None):
        client.reset()


def CarState(client):
    State = client.getCarState()
    pos = State.kinematics_estimated.position.to_numpy_array()
    velocity = State.speed
    kinematics_estimated = State.kinematics_estimated.position
    return pos, velocity, kinematics_estimated


def SimMove(t,s,client):
    car_controls.throttle = t
    car_controls.steering = s
    client.setCarControls(car_controls)
    time.sleep(delta)

def A(kP,kD,kI,client):
    # Установили параметры
    speed_controller.setControllerParams(kP, kD, kI)
    _,v0,_=CarState(client)
    # Получаем steering и throttle
    e_velocity_p = velocity_control(v0, min_velocity, max_velocity)
    correction_v = speed_controller.PIDController(e_velocity_p)
    throttle = client.getCarControls().throttle * (1 + correction_v)
    if throttle > 1:
        throttle = 1
    steering = client.getCarControls().steering
    #TODO Доделать Twiddle алгоритм для контролерра скорости



def Twiddle(kP,kD,kI):
    # Choose an initialization parameter vector
    p = [kP,kD,kI]
    # Define potential changes
    dp = [1, 1, 1]
    # Calculate the error
    best_err = Sim(p)

    threshold = 0.001

    while sum(dp) > threshold:

        for i in range(len(p)):
            p[i] += dp[i]
            err = Sim(p)

            if err < best_err:  # There was some improvement
                best_err = err
                dp[i] *= 1.1
            else:  # There was no improvement
                p[i] -= 2 * dp[i]  # Go into the other direction
                err = Sim(p)

                if err < best_err:  # There was an improvement
                    best_err = err
                    dp[i] *= 1.05
                else:  # There was no improvement
                    p[i] += dp[i]
                    # As there was no improvement, the step size in either
                    # direction, the step size might simply be too big.
                    dp[i] *= 0.95

# TODO Добавить генератор трасс и получить его нормали. Для визуализации можно использовать просто matplotlib,
#  а UE4 использовать только для демонтрации проекта
# generate_track()
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
# car_controls = airsim.CarControls()
d_vec = direct_vector(client, "PhysXCar")
print(d_vec / np.linalg.norm(d_vec, ord=2))
# move_to_point_direct(client, "PhysXCar")
# Бесконечный цикл?

# Параметры симулятора
delta = 0.01

# PID
kp_curve = 0
kd_curve = 0
ki_curve = 0

kp_velocity = 0
kd_velocity = 0
ki_velocity = 0
max_velocity = 15
min_velocity = 10
is_velocity_control = False
v0 = 6
best_error_v_p = velocity_control(0,min_velocity,max_velocity)
home_pose = client.getCarState().kinematics_estimated.position
car_controls = airsim.CarControls()
car_controls.throttle = 1
car_controls.steering = 0
# print(f"TruePose",client.simGetTrueVehiclePose("PhysXCar").position)
speed_controller = Controller.Controller()
curve_controller = Controller.Controller()
car_controls = client.getCarControls("PhysXCar")
is_curve_control = False

t0 = time.time()


""""
linear_accel = []
ang_accel = []
accel = []
positions = np.array([[client.getCarState().kinematics_estimated.position.to_numpy_array()[0],
                      client.getCarState().kinematics_estimated.position.to_numpy_array()[1]]])
velocity = np.array([[0,client.getCarState().speed]])
trottles = np.array([[0,car_controls.throttle]])
steerings = np.array([[0,car_controls.steering]])
error_lenght_prev,_ =SR.distance_to_point(client.getCarState().kinematics_estimated.position.to_numpy_array())
errors_lenght = np.array([[error_lenght_prev]])
"""
while True:

    posAgent, speedAgent = CarState(client)
    if client.simGetCollisionInfo().has_collided:
        reset_environment(client)

    if not client.simIsPause():
        SimMove(throttle,steering)
        Twiddle(kp_velocity, kd_velocity, ki_velocity)
        if is_velocity_control:
            # velocity = np.append(velocity,[[time.time() - t0,speedAgent]],axis=0)
            # trottles = np.append(trottles, [[time.time() - t0, car_controls.throttle]], axis=0)

            # print(f"error={error}, throttle={throttle}, time={time.time()}")
            # print(velocity)




# get camera images from the car
# responses = client.simGetImages([
#    airsim.ImageRequest(0, airsim.ImageType.Infrared),
#    airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)])
# print('Retrieved images: %d', len(responses))

# do something with images
# for response in responses:
#    if response.pixels_as_float:
#        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#        airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
#    else:
#        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#        airsim.write_file('py1.png', response.image_data_uint8)
"""
        if False:
            error_lenght,p =SR.distance_to_point(posAgent)
            errors_lenght = np.append(errors_lenght,[error_lenght])
            p = np.array([p[0],SR.get_function()(p)[0]])
            print(f"posAgent{[posAgent]}")
            print(f"p{p}")
            sign = sign_steering(posAgent,p,SR.get_function())
            print(f"error_lenght={error_lenght},sign={sign}")
            print(f"error_lenght={error_lenght_prev}")
            correction = sign*(kp*error_lenght + kd*((error_lenght - error_lenght_prev)/delta) + ki*np.sum(errors_lenght)*delta)
            if np.abs(correction[0]) >=1:
                correction = [np.sign(correction[0])]
            steering =correction
            print(f"correction{correction}")
            print(f"steering{steering}")
            car_controls.steering = steering[0]
            error_lenght_prev = error_lenght
            if errors_lenght.size>100:
                errors_lenght = np.array([[error_lenght]])
    steerings = np.append(steerings, [[time.time() - t0, car_controls.steering]], axis=0)
    positions = np.append(positions,[[posAgent[0],posAgent[1]]],axis=0)
    if False:
        client.simContinueForTime(20)
        plt.plot(linear_accel,"o",c='Red')
        plt.show()
        plt.plot(ang_accel,"o", c='Red')
        plt.show()
        plt.plot(np.sqrt(np.array(linear_accel)**2+np.array(ang_accel)**2),"-o", c='Red')
        plt.show()
        linear_accel = []
        ang_accel = []
    if time.time() - t0 > 10000000000000:
        t0 = time.time()
        client.simPause(True)
        client.simPrintLogMessage("Pause")
        plot_g(positions)
        print(f"Прошло 60 секунд")
        client.simPause(False)
if is_curve_control:
    error_lenght, p = SR.distance_to_point(posAgent)
    errors_lenght = np.append(errors_lenght, [error_lenght])
    p = np.array([p[0], SR.get_function()(p)[0]])
    sign = sign_steering(posAgent, p, SR.get_function())
    correction = sign * (kp * error_lenght + kd * ((error_lenght - error_lenght_prev) / delta) + ki * np.sum(
        errors_lenght) * delta)
    steering = sign * error_lenght[0]
    car_controls.steering = kp * steering
    error_lenght_prev = error_lenght
    if errors_lenght.size > 100:
        errors_lenght = np.array([[error_lenght]])"""