cmake_minimum_required(VERSION 3.0.2)
project(tayo_perception)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_geometry
    laser_geometry
    pcl_ros
    roscpp
    sensor_msgs
    std_msgs
    image_transport
    # trackmanagement_lib
    message_generation
    autoware_msgs
    vehicle_msgs
    #ad_challenge_msgs
    obstacle_msgs
    novatel_gps_msgs
    decision_msgs
    )

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    #  LIBRARIES targetless
    CATKIN_DEPENDS cv_bridge image_geometry pcl_ros roscpp sensor_msgs std_msgs image_transport message_runtime obstacle_msgs decision_msgs
    #  DEPENDS system_lib
    )

include_directories(include)
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    # ${trackmanagement_lib_INCLUDE_DIRS}
    )

if(Eigen3_FOUND)
    set(CUSTOM_EIGEN ${EIGEN3_INCLUDE_DIR})
else()
    find_package(Eigen)
    set(CUSTOM_EIGEN ${EIGEN_INCLUDE_DIR})
endif()
include_directories(
        ${CUSTOM_EIGEN}
)

file(GLOB_RECURSE project_HEADERS
    "*.h"
    "*.hpp"
    )

add_executable(perception_node
    ${project_HEADERS}
    src/perception/tayoPerception_node.cpp
    src/perception/tayoPerception.cpp
    src/perception/KalmanFilter.cpp
    src/perception/tayoAssocation.cpp
    src/perception/tayoKalmanFilter.cpp
    )
# add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
add_dependencies(perception_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(perception_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    )

#add_executable(perception_qual_node
#    ${project_HEADERS}
 #   src/perception_qualification/perception_qual_node.cpp
  #  src/perception_qualification/perception_qual.cpp
   # )
    
# add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
#add_dependencies(perception_qual_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(perception_qual_node
#    ${catkin_LIBRARIES}
#    ${OpenCV_LIBS}
#    Eigen3::Eigen
 #   )

#add_executable(tracking_node
#    ${project_HEADERS}
#    src/tracking/tracking.cpp
#    src/tracking/tracking_node.cpp
#    src/tracking/tayo_tracking.cpp
#    )

#add_dependencies(tracking_node ${catkin_EXPORTED_TARGETS})

#target_link_libraries(tracking_node
#    ${catkin_LIBRARIES}
#    ${OpenCV_LIBS}
#    Eigen3::Eigen
#    )
