import CurveController
import SplineRoad
import os

FR = SplineRoad.FunctionRoad()
FR.generate_data()
FR.plot_cones()
FR.move_data()
print(os.listdir("d:/Users/user/Documents/AirSim"))
