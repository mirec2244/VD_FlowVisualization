#pragma once
#include <opencv2/opencv.hpp>

#pragma once
class FlowVisualization
{
public:
	FlowVisualization(int precision, float step_delay, int flow_size, int num_of_points);
	void reset();
	void run();
	void key_pressed(int key, int i);
	void curl(cv::Mat& input, cv::Mat& output, cv::Mat& output_color);
	static std::vector<cv::Vec2f> move_points_by_range_kutta(cv::Mat& flow, std::vector<cv::Vec2f> points, float differential);
	void redraw_flow(cv::Mat& input);
private:
	const int width_flow_ = 128, height_flow_ = 128, width_ = 512, height_ = 512;
	int frame_ = 0, precision_, flow_size_, num_of_points_;
	float step_delay_;
	cv::Mat flow_, with_curl_, with_curl_color_, rotation_matrix_, speed_matrix_;
	std::vector<cv::Vec2f> points_;
	cv::Point2f scale_factor_;
};
