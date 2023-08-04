#pragma once 
#include "TensorflowModelParser.h"
#include <opencv2/core.hpp>
#include "Sku110KStructures.h"

class Sku110KModelInferencer {
public:
using Data=emmerger::Data;
using DataWithAvgScore=emmerger::DataWithAvgScore;
Sku110KModelInferencer() = default;
~Sku110KModelInferencer() = default;
Sku110KModelInferencer(const std::string& model_path, const float hard_score_rate = 0.5f, const float hard_score_th = 0.1):
model_path_(model_path),hard_score_rate_(hard_score_rate), hard_score_threshold_(hard_score_th){};
bool init(){
    return initModel();
};

void preprocess(const cv::Mat& rgb_img);
bool run();
void showBoxes(const std::string& save_fn);

private:
float hard_score_rate_ = 0.5f;
float hard_score_threshold_ = 0.1f;
float score_threshold_ = 0.5f;
std::string model_path_;
std::string gpu_visible_device_="0";
float gpu_memory_fraction_ = 0.4f; 
TensorflowModelParser parser_;
//std::vector<std::string> input_layers_{"input_4:0"};
//std::vector<std::string> output_layers_{"filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0", "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0",
//"filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0","filtered_detections/map/TensorArrayStack_3/TensorArrayGatherV3:0"};

std::vector<std::string> input_layers_{"input_4"};
std::vector<std::string> output_layers_{"filtered_detections/map/TensorArrayStack/TensorArrayGatherV3", "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3",
"filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3","filtered_detections/map/TensorArrayStack_3/TensorArrayGatherV3"};

const int min_side_=800;
const int max_side_=1333;
const int max_detections_ = 9999;
float scale_ = 1.0f;
std::vector<tensorflow::Tensor> input_tensor_arr_;
std::vector<tensorflow::Tensor> output_tensor_arr_;
std::vector<Data> output_datas_;
cv::Mat img_;
std::vector<DataWithAvgScore> filtered_datas_;

private:
bool initModel(){
    input_tensor_arr_.clear();
    output_tensor_arr_.clear();        
    return parser_.Init(model_path_, output_layers_, input_layers_, gpu_visible_device_, gpu_memory_fraction_);
};
cv::Mat resizeImg(const cv::Mat& img);
//substract mean img's BGR from imageNet 
cv::Mat subMeanBGR(const cv::Mat& img);
void warpImgToTensor(const cv::Mat& img);
void warpOutputTensorToData();

};