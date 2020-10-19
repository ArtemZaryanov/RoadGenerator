# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import airsim.utils
import numpy as np
import SplineRoad as SR
import CurveController as Controller
import matplotlib.pyplot as plt
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


# Генерация трассы
def generate_track(return_function = False):
    FR = SR.FunctionRoad()
    FR.generate_data()
    if return_function:
        return SR.get_function(), _,_
    else:
        return _,_

# TODO Добавить генератор трасс и получить его нормали. Для визуализации можно использовать просто matplotlib,
#  а UE4 использовать только для демонтрации проекта

generate_track()
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
car_controls = airsim.CarControls()
car_controls.throttle = 1
car_controls.steering = 0
# print(f"TruePose",client.simGetTrueVehiclePose("PhysXCar").position)
PID_Controller = Controller.Controller(0.1)
car_controls = client.getCarControls("PhysXCar")
v0 = 20
k = 0
is_velocity_control = True
is_curve_control = True
max_velocity = 30
min_velocity = 25
t0 = time.time()
linear_accel = []
ang_accel = []
accel = []
velocity = np.array([[0,client.getCarState().speed]])
trottles = np.array([[0,car_controls.throttle]])
while True:
    # get state of the car
    # car_state = client.getCarState()
    # car_state.kinematics_estimated.angular_velocity.get_length()
    # print("Line_accel %d",car_state.kinematics_estimated.linear_acceleration)
    # print("Ang_accel %d ",car_state.kinematics_estimated.angular_acceleration)
    # print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    # print(np.degrees(airsim.to_eularian_angles(client.simGetObjectPose("PhysXCar").orientation)))
    # print(client.simGetObjectPose("PhysXCar").position)
    if not client.simIsPause():
        client.setCarControls(car_controls)
        time.sleep(0.1)
        if is_velocity_control:
            error = velocity_control(client.getCarState().speed,min_velocity,max_velocity)
            _, throttle = PID_Controller.PIDController(0,client.getCarControls().throttle,error)
            if throttle>1:
                throttle = 1

            car_controls.throttle = throttle
            car_controls.steering = 0
            velocity = np.append(velocity,[[time.time() - t0,client.getCarState().speed]],axis=0)
            trottles = np.append(trottles, [[time.time() - t0, car_controls.throttle]], axis=0)

            print(f"error={error}, throttle={throttle}, time={time.time()}")
            print(f"error_lenght={0}")
            print(velocity)
        if is_curve_control:
            error_lenght = C
            pass

    # print(f"Line_accel {car_state.kinematics_estimated.linear_acceleration}")
    # print(f"Ang_accel {car_state.kinematics_estimated.angular_acceleration.get_length()}")
    # linear_accel.append(car_state.kinematics_estimated.linear_acceleration.get_length())
    # ang_accel.append(car_state.kinematics_estimated.angular_acceleration.get_length())
    # car_controls.steering = int(np.round(s))
    # print("Line_velosity %d", car_state.kinematics_estimated.linear_velocity.get_length())
    # print("Ang_velosit %d", car_state.kinematics_estimated.angular_velocity.get_length())
    # if car_state.kinematics_estimated.angular_velocity.get_length() != 0:
    #    print("Radius %d",
    #          car_state.kinematics_estimated.linear_velocity.get_length() / car_state.kinematics_estimated.angular_velocity.get_length())
    if False:
        k=0
        client.simContinueForTime(20)
        plt.plot(linear_accel,"o",c='Red')
        plt.show()
        plt.plot(ang_accel,"o", c='Red')
        plt.show()
        plt.plot(np.sqrt(np.array(linear_accel)**2+np.array(ang_accel)**2),"-o", c='Red')
        plt.show()
        linear_accel = []
        ang_accel = []
    k=k+1
    if time.time() - t0 > 60:
        t0 = time.time()
        client.simPause(True)
        # Рисуем графики Отдельо создание figure и прочее. Чтобы была просто отправка и отрсиовка данных без
        # именований на ходу(может лучше будет
        # Продолжать с того же времени. Значит нудны две переменныхе. Одна для отчета. Друга 60*i, i++
        # Нарисовать прямые. Коридор допустимых скоростей
        fig,axes = plt.subplots(2,1)
        axes[0].plot(velocity[:,0],velocity[:,1])
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
        plt.show()
        #Имя
        print(f"Прошло 60 секунд")




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