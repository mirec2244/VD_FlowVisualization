#include "FlowVisualization.h"

int main()
{
	auto flow_visualization = new FlowVisualization(50, 0.05, 30, 100);
	flow_visualization->run();
	return 0;
}
