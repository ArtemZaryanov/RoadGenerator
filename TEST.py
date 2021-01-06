import CurveController
import SplineRoad
import SplineRoadNew
import os
from tensorflow import keras


SR = SplineRoadNew.SplineRoad()
SR.generate_data()
SR.plot_cones(plot_func=False)
SR.move_data()
# FR = SplineRoad.FunctionRoad()
# FR.generate_data()
# FR.plot_cones()
# FR.move_data()
