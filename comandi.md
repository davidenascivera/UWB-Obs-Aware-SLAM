# COMANDI
## AVVIARE LA SIMULAZIONE

1. Avviare il MicroAgent (puoi non chiuderlo mai)
```bash
MicroXRCEAgent udp4 -p 8888
```
2. Avviare Qground control (non serve chiudere neanche questo)
```bash
../QGroundControl-x86_64.AppImage 
```

3. Avviare la simulazione con gazebo
```bash
cd ../PX4-Autopilot/ && make px4_sitl gz_x500
```

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /model/x500_0/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry
```

## Compilare con colcon build:
```bash
cd ~/UV_proj && colcon build --packages-select tutorial_px --symlink-install
```


----------------------------------------------------------
## Come registrare con rosbag e riprodurre

Per registrare i topic che ci servono apriamo un **nuovo terminale** (dopo aver fatto i `source` giusti):

```bash
# Se servono anche i messaggi custom (px4_msgs ecc.)
source /opt/ros/humble/setup.bash
source ~/UV_proj/install/setup.bash

# Registra i topic (incluso magnetometro)
ros2 bag record \
  /world/default/model/x500_0/link/base_link/sensor/imu_sensor/imu \
  /fmu/out/sensor_combined \
  /fmu/out/sensor_mag \
  /fmu/out/vehicle_magnetometer \
  /model/x500_0/odometry \
  -o drone_data_bag_magn
```
Per riprodurre la rosbag registrata:
```bash
ros2 bag play drone_data_bag/
```
Per info sul contenuto:
```bash
ros2 bag info <nomeCartella>
```

### Per riprodurre dopo la rosbag
primo terminale (fa andare la rosbag)
```bash
source /opt/ros/humble/setup.bash
source ~/UV_proj/install/setup.bash
ros2 bag play drone_data_bag_magn --loop
```
secondo terminale (ascoltare con ros2 topic echo)
```bash
source /opt/ros/humble/setup.bash
source ~/UV_proj/install/setup.bash

# vedi i topic mentre il player è in esecuzione
ros2 topic list -t

# echo con QoS da sensori (PX4 pubblica best_effort)
ros2 topic echo /fmu/out/sensor_combined --qos-profile sensor_data

# echo magnetometer topics
ros2 topic echo /fmu/out/sensor_mag --qos-profile sensor_data
ros2 topic echo /fmu/out/vehicle_magnetometer --qos-profile sensor_data
```
## Aprire plotjuggler
```bash
ros2 run plotjuggler plotjuggler
```

## Comandi settati nel MIO VSCODE per terminale

- Creare nuovo terminale: `ctrl+shift+ò`  
- Gruppo precedente: `ctrl+shift+à`  
- Gruppo successivo: `ctrl+shift+ù`  
- Gruppo successivo nel gruppo: `alt` + frecce  
- Splitta terminale: `ctrl+shift+5`  
- Focus terminale (senza toggle): `ctrl+ò`  
