cmake_minimum_required(VERSION 3.0.2)
project(obstacle_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  visualization_msgs
)

 add_message_files(
   FILES
   obstacle.msg
   obstacleArray.msg
 )

 generate_messages(
   DEPENDENCIES
   std_msgs
   visualization_msgs
 )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_msg
  CATKIN_DEPENDS std_msgs visualization_msgs message_runtime
#  DEPENDS system_lib
)

