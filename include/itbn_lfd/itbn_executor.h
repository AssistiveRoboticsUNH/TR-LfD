/*
Madison Clark-Turner
10/14/2017
*/

/*
 receives the run command.
 passes the observations into the parser and receives information
 from the model
*/

#ifndef ITBNExecutor_H
#define ITBNExecutor_H

#include "itbn_lfd/ITBNGetNextAction.h"
#include <iostream>

#include "executor.h"

class ITBNExecutor: public Executor{
private:
	// service names
	std::string srv_nextact_name = "get_next_action";

	// subscribers
	ros::ServiceClient srv_nextact;

	// services calls
	int srvNextAct(int act);
	int getNextAct(int act);

public:
	ITBNExecutor(ros::NodeHandle);
};

#endif