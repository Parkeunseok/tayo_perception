#include <iostream>
#include <ros/ros.h>

#include "perception/tayoPerception.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "perception_node");
    ros::NodeHandle nh;

    Perception::tayoPerception tayoPerception_node(nh);
    ros::spin();
    return 0;
}
