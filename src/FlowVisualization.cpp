#include "FlowVisualization.h"
#include <ctime>
#include <random>
#include <string>
#include <cmath>

// Constructor
FlowVisualization::FlowVisualization(const int precision, const float step_delay, const int flow_size, const int num_of_points)
{
	srand(time(NULL));
	cv::namedWindow("Speed");
	cv::namedWindow("Flow Visualization");
	cv::namedWindow("Rotation Visualization");
	this->precision_ = precision;
	this->step_delay_ = step_delay;
	this->flow_size_ = flow_size;
	this->num_of_points_ = num_of_points;
	scale_factor_ = cv::Vec2f(width_ / static_cast<float>(width_flow_), height_ / static_cast<float>(height_flow_));
	with_curl_ = cv::Mat(height_flow_, width_flow_, CV_32F);
	reset();
}


// Reset all points and generate new points in range
void FlowVisualization::reset()
{
	points_.clear();
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_int_distribution<int> distribution(0, 128);
	for (size_t i = 0; i < num_of_points_; i++)
	{
		points_.emplace_back(distribution(generator), distribution(generator));
	}
}

// Run flow visualization - get data and redraw
void FlowVisualization::run()
{
	cv::Mat storage_mat;
	size_t i = 0;
	
	while(true)
	{
		if (i < flow_size_) {
			char file_name[100];
			snprintf(file_name, sizeof(file_name), "dataflow/u%05d.yml", i);
			cv::FileStorage storage(file_name, cv::FileStorage::Mode::FORMAT_AUTO | cv::FileStorage::Mode::READ);
			storage["flow"] >> storage_mat;
			storage.release();
			i++;
		}

		
		frame_++;
		redraw_flow(storage_mat);
		key_pressed(cv::waitKey(30), i);
	}
}

// Check if key was pressed - Quit (q), Pause (space), Reset (r), Increase step delay (W), Decrease step delay (s), Precision+ (d), Precision- (a)
void FlowVisualization::key_pressed(const int key, int i)
{
	switch(key)
	{
		case ' ':
			cv::waitKey(0);
			break;

		case 'q':
			exit(0);

		case 'r':
			i = 0;
			reset();
		
		case 'w':
			step_delay_ *= 1.1;
			break;
		
		case 's':
			step_delay_ /= 1.1;
			break;
		
		case 'd':
			precision_ += 5;
			break;
		
		case 'a':
			precision_ += 5;
			break;

		default:
			break;
	}
}

// Get mats of curl
void FlowVisualization::curl(cv::Mat& input, cv::Mat& output, cv::Mat& output_color)
{
	for (auto y = 0; y < height_flow_; y++)
	{
		for (auto x = 0; x < width_flow_; x++)
		{
			if (x > 0 && y > 0 && x < width_flow_ - 1 && y < height_flow_ - 1)
				output.at<float>(y, x) = (input.at<cv::Point2f>(y, x - 1).y - input.at<cv::Point2f>(y, x + 1).y) - (input.at<cv::Point2f>(y - 1, x).x - input.at<cv::Point2f>(y + 1, x).x);
			else
				output.at<float>(y, x) = 0;
		}
	}

	auto normalized = cv::Mat(output.rows, output.cols, CV_32FC1);
	cv::normalize(output, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	normalized.convertTo(normalized, CV_8UC1);
	applyColorMap(normalized, output_color, cv::COLORMAP_JET);
	resize(output_color, rotation_matrix_, cv::Size(), width_ / width_flow_, height_ / height_flow_, cv::INTER_CUBIC);
}

// Get k value
cv::Vec2f remap_and_get_k_value(const cv::Mat& flow, cv::Point2f point)
{
	cv::Mat remap;
	cv::remap(flow, remap, cv::Mat(1, 1, CV_32FC2, &point), cv::noArray(),
		cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
	return remap.at<cv::Vec2f>(0, 0);
}

// Move points - use range kutta
auto FlowVisualization::move_points_by_range_kutta(cv::Mat& flow, std::vector<cv::Vec2f> points,
                                                   const float differential) -> std::vector<cv::Vec2f>
{
	std::vector<cv::Vec2f> new_points;
	for (const auto& i : points)
	{
		const auto k1 = remap_and_get_k_value(flow, i) * differential;
		const auto k2 = remap_and_get_k_value(flow, i + k1 * 0.5f) * differential;
		const auto k3 = remap_and_get_k_value(flow, i + k2 * 0.5f) * differential;
		const auto k4 = remap_and_get_k_value(flow, i + k3) * differential;
		new_points.push_back(i + ((1. / 6) * (k1 + 2 * k2 + 2 * k3 + k4)));
	}
	return new_points;
}

// Redraw flow and all windows
void FlowVisualization::redraw_flow(cv::Mat& input)
{
	double min, max;
	minMaxLoc(input, &min, &max);
	const float arrow_scale = MAX(abs(min), abs(max)) / MIN(scale_factor_.x, scale_factor_.y);
	curl(input, with_curl_, with_curl_color_);
	resize(with_curl_, flow_, cv::Size(), width_ / width_flow_, height_ / height_flow_, cv::INTER_CUBIC);
	cv::cvtColor(flow_, flow_, cv::COLOR_GRAY2BGR);
	speed_matrix_ = flow_.clone();
	
	for (auto y = 0; y < height_flow_; y++)
	{
		for (auto x = 0; x < width_flow_; x++)
		{
			cv::Point2f pos((0.5f + x) * scale_factor_.x, (0.5f + y) * scale_factor_.y);
			auto start = pos - input.at<cv::Point2f>(y, x) * arrow_scale;
			auto end = pos + input.at<cv::Point2f>(y, x) * arrow_scale;
			if (x > 0 && y > 0 && x < 127 && y < 127) {
				auto color = with_curl_color_.at<cv::Vec3b>(y - 1, x - 1);
			}
		}
	}

	
	for (size_t i = 0; i < precision_; i++)
	{
		auto new_points = move_points_by_range_kutta(input, points_, step_delay_);
		for (size_t j = 0; j < points_.size(); j++)
		{
			auto point_new = new_points[j].mul(scale_factor_);
			auto point_old = points_[j].mul(scale_factor_);
			arrowedLine(flow_, cv::Point(point_old[0], point_old[1]), cv::Point(point_new[0], point_new[1]), cv::Scalar(255, 0, 255), 2);
			arrowedLine(rotation_matrix_, cv::Point(point_old[0], point_old[1]), cv::Point(point_new[0], point_new[1]), cv::Scalar(255, 0, 255), 2);

			const auto speed = sqrt(pow(point_new[0] - point_old[0], 2) + pow(point_new[1] - point_old[1], 2)) * 100;
			cv::putText(speed_matrix_, std::to_string((int)speed), cv::Point(point_new[0], point_new[1]),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 255), 1);
		}
		points_ = new_points;
	}

	cv::putText(rotation_matrix_, "frame: " + std::to_string(frame_), cv::Point(10, 20),
	            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(10, 10, 10), 1, cv::LINE_AA);

	//cv::imshow("Speed", speed_matrix_);
	cv::imshow("Flow Visualization", flow_);
	cv::imshow("Rotation Visualization", rotation_matrix_);
}