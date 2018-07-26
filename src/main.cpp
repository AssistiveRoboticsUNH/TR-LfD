/*
main.cpp
Madison Clark-Turner
10/14/2017
*/

#include <ros/ros.h>
#include <ros/package.h>
#include <string>
#include <iostream>
#include "../include/itbn_lfd/itbn_executor.h"

int main(int argc, char** argv){
	ros::init(argc, argv, "itbn");
	ros::NodeHandle n;
	ITBNExecutor itbn(n);
	ROS_INFO("ITBN Executor ready");
	ros::spin();
}