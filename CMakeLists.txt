cmake_minimum_required(VERSION 2.8.3)
project(itbn_lfd)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  roslib
  rosbag
  std_srvs
  nao_msgs
  message_generation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

###########
## Build ##
###########

add_service_files(
  FILES
  ITBNGetNextAction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

#################
## Executables ##
#################

add_executable( itbn_executor_v2 src/main.cpp src/itbn_executor.cpp src/executor.cpp)
target_link_libraries(itbn_executor_v2 ${catkin_LIBRARIES})
add_dependencies( itbn_executor_v2 ${PROJECT_NAME}_generate_messages_cpp )
