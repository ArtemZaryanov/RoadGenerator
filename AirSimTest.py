# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import numpy as np


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
    return pos1 - pos0


# TODO Добавить генератор трасс и получить его нормали. Для визуализации можно использовать просто matplotlib,
#  а UE4 использовать только для демонтрации проекта


# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()

client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
# car_controls = airsim.CarControls()
d_vec = direct_vector(client, "PhysXCar")
print(d_vec / np.linalg.norm(d_vec, ord=2))
# Бесконечный цикл?
# while True:
# get state of the car
# car_state = client.getCarState()
# print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
# print(np.degrees(airsim.to_eularian_angles(client.simGetObjectPose("PhysXCar").orientation)))
# print(client.simGetObjectPose("PhysXCar").position)
# set the controls for car
# car_controls.throttle = 0
# car_controls.steering = 1
# car_controls.manual_gear = 2
# car_controls.handbrake =
# client.setCarControls(car_controls)
# Ждем когда машина проедет
# time.sleep(1)

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
