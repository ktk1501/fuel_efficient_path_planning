# fuel_efficient_path_planning

# Dependencies
- numpy
- matplotlib
- opencv-python
- mpl_toolkits

# How to run
```bash
python frenet_optimal_trajectory.py    (for 3D simulation)
python frenet_optimal_trajectory_2d.py (for 2D simulation)
```
# Results
## 2D simulation
![fuel_test_1](https://user-images.githubusercontent.com/40379815/100630507-f7fdf680-336d-11eb-8c6d-978caa96cf2a.gif)
## 3D simulation
![plot2_integrated](https://user-images.githubusercontent.com/40379815/100629864-23ccac80-336d-11eb-8d4f-3672e1413d82.gif)
## Fuel comsumption difference during 3D simulation
It only considers stationary_distance/time, acceleration_distance/time and terrain slope. Which are considered to be the most significant factors in fuel efficiency.
![스크린샷 2020-09-22 오전 4 51 33](https://user-images.githubusercontent.com/40379815/100629813-157e9080-336d-11eb-807e-31121649bea2.png)
