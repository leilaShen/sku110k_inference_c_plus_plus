#pragma once
#include <opencv2/core.hpp>
#include "Sku110KStructures.h"
#include <iostream>
#include <iterator>

namespace emmerger{

cv::Mat gaussianBlur(const int w, const int h);
cv::Mat aggregateGaussians(const std::vector<Result>& result_arr, const cv::Size& heat_map_size);

std::tuple<std::vector<cv::Mat>,cv::Mat, int, cv::Mat> removeRedundant(const cv::Rect& bbox,const int k, const cv::Mat& image, const cv::Mat sub_heat_map, std::vector<cv::Mat>& cov, cv::Mat& mu);

void setCandidates(std::vector<Candidate>& candidates, const std::vector<cv::Mat>& cov, const cv::Mat heat_map, const cv::Mat& mu, const int num,const cv::Point& offset, const cv::Mat& sub_heat_map);

std::vector<DataWithAvgScore> filterDuplicateCandidates(const std::vector<Data>& data, const cv::Mat& image);

}