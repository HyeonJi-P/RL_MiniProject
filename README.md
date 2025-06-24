# RL_MiniProject

# Environment setup
## 1. Docker 셋팅
`git clone https://github.com/ryomo/ros2-gazebo-docker.git`
* docker-compose.yml 파일 내용을 현재 레포지토리의 **cham_docker-compose.yml** 내용으로 변경 후 레포지토리의 안내에 따라 도커 생성 및 접속

## 2. clone repositories into workspace
```
cd ~  
git clone https://github.com/HyeonJi-P/RL_MiniProject.git
cp RL_MiniProject/project .
cd project/dev_ws/src
# src 내 세부 turtlebot3 관련 레포지토리는 기본적으로 turtlebot3 예제의 설정을 따름 : https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/

```
* 최종 tree구조
```
project/ 
 ├─ dev_ws/
 │   └─ src/
 │       ├─ chamoe_rl/   # <‑‑ main package
 │       └─ turtlebot3/
 │       └─ turtlebot3_machine_learning/
 │       └─ turtlebot3_msg/
 │       └─ turtlebot3_simulations/
 └─ docker/
     └─ ros2/docker‑compose.yml
```

## 3. model, urdf, world file 위치 변경
* 레포지토리에서 제공한 파일(각각 model, urdf, world폴더 내에 있는 것들)
* /home/dockeruser/project/dev_ws/src/turtlebot3_simulations/turtlebot3_gazebo 아래에 있는 model, urdf, world 디렉토리 내에  넣기
* World파일의 경우 real_time_factor에서 시뮬레이션 속도 조절가능
* 현재 turtlebot3_burgur_cam 모델에 depthcamera 추가된 버전

## 4. YOLOv11 install + weight (inside container later)
`pip install ultralytics `
* y11n_gz_chamoebest.pt
`wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MVPNiJVxUNUAMaefo5tG2Dmo0LkZX-dg' -O y11n_gz_chamoebest.pt`

## 5. Docker의 bashrc 설정
```
source /opt/ros/humble/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=/home/dockeruser/dev_ws
export ROS_DOMAIN_ID=11
export TURTLEBOT3_MODEL=burger_realsense
alias pj='cd ~/project/dev_ws'
alias bd='colcon build --symlink-install'
alias es='source $HOME/project/dev_ws/install/setup.bash'
```

## 6. cd ~/dev_ws
colcon build --symlink-install
source install/setup.bash


## Build
`colcon build`

# Run
* Gazebo world + TB3 도커 내 각각의 터미널에서 실행. 학습 혹은 테스트시 아래 4개 실행 필수
`ros2 launch turtlebot3_gazebo RL_project_chamoe.launch.py`
`ros2 run chamoe_rl chamoe_rl_RK_environment`
`ros2 run chamoe_rl chamoe_rl_gazebo_interface`
`ros2 run chamoe_rl chamoe_detect YOLOv11 detection node`

* DQN 학습
* `python3 /home/dockeruser/project/dev_ws/src/chamoe_rl/chamoe_rl/DQN_agent.py 1 {에피소드수} `

* Q-Learning 학습
* `python3 /home/dockeruser/project/dev_ws/src/chamoe_rl/chamoe_rl/Q_agent.py 1 {에피소드수} `
  
* (옵션) 액션 모니터
`ros2 run chamoe_rl chamoe_rl_6action_graph`

* (옵션) 보상 그래프
`ros2 run chamoe_rl chamoe_rl_result_graph`

# Test 
* DQN : 학습된 파일명에 따라 코드에서 load내용 수정 필요
`python3 /home/dockeruser/project/dev_ws/src/chamoe_rl/chamoe_rl/DQN_test.py`
`python3 /home/dockeruser/project/dev_ws/src/chamoe_rl/chamoe_rl/Q_test.py`

