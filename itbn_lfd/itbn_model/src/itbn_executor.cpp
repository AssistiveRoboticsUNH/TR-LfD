/*
Madison Clark-Turner
10/14/2017
*/

#include "../include/itbn_lfd/itbn_executor.h"

using namespace itbn_lfd;

ITBNExecutor::ITBNExecutor(ros::NodeHandle node): Executor(node)
{	
	srv_nextact = n.serviceClient<ITBNGetNextAction>(srv_nextact_name);
}

int ITBNExecutor::srvNextAct(int act){
	ITBNGetNextAction srv;
	srv.request.last_act = act;
	int nextact = -1;
	if (srv_nextact.call(srv)){
		ROS_INFO("Call to service: %s, succesful!", srv_nextact_name.c_str());
		nextact = srv.response.next_act;
	}
	else{
		ROS_INFO("Call to service: %s, failed.", srv_nextact_name.c_str());
	}

	return nextact;
}

int ITBNExecutor::getNextAct(int act){
	int nextact = -1;

	nextact = srvNextAct(act);

	return nextact;
}