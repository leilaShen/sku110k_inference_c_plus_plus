#include "Sku110KModelInferencer.h"
#include "EMMerger.h"
#include <opencv2/imgproc.hpp>
#include <sstream>


cv::Mat Sku110KModelInferencer::resizeImg(const cv::Mat& img){
    int rows = img.rows;
    int cols = img.cols;
    int smallest_side = std::min(rows, cols);
    this->scale_ = static_cast<float>(min_side_)/static_cast<float>(smallest_side);
    int largest_side = std::max(rows, cols);
    if (largest_side * this->scale_ > max_side_) {
        this->scale_ = max_side_/largest_side;
    }
    cv::Mat dst_img;
    cv::resize(img, dst_img, cv::Size(static_cast<int>(this->scale_*cols), static_cast<int>(this->scale_*rows)), this->scale_, this->scale_);
    return dst_img;
}

cv::Mat Sku110KModelInferencer::subMeanBGR(const cv::Mat& img){
    cv::Mat mean_bgr = cv::Mat(img.size(), CV_32FC3, cv::Scalar(103.939,116.779,123.68));
    cv::Mat f_img;
    img.convertTo(f_img, CV_32FC3);
    f_img = f_img - mean_bgr;
    return f_img;
}

void Sku110KModelInferencer::preprocess(const cv::Mat& rgb_img){
    auto f_img = this->subMeanBGR(rgb_img);
    auto preprocessed_img = this->resizeImg(f_img);
    this->warpImgToTensor(preprocessed_img);
    this->img_ = rgb_img.clone();
}


void Sku110KModelInferencer::warpImgToTensor(const cv::Mat& img){  
    // const int depth = 3;
    // const float* source_data = (float*)img.data;
    // int height = img.rows;
    // int width = img.cols;
    // tensorflow::Tensor input_tensor(DT_FLOAT, tensorflow::TensorShape({1,height,width,3}));
    // auto input = input_tensor.tensor<float,4>();
    // for(int y = 0; y < height; y++){
    //     const float* source_row = source_data + (y*width*depth);
    //     for(int x = 0; x < width; x++){
    //         const float* source_pixel = source_row + (x*depth);
    //         for (int c = 0; c < depth; ++c){
    //             const float* source_value = source_pixel + c;
    //             input(0,y,x,c) = *source_value;
    //         }
    //     }
    // }
    // this->input_tensor_arr_.push_back(input_tensor);
    ////////////////////////////////below is a new method to warp img to tensor////////////////////////////
    // allocate a Tensor
    tensorflow::Tensor input_img(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,img.rows,img.cols,3}));

    // get pointer to memory for that Tensor
    float *p = input_img.flat<float>().data();
    // create a "fake" cv::Mat from it 
    cv::Mat camera_img(img.rows, img.cols, CV_32FC3, p);

    // use it here as a destination
    img.convertTo(camera_img, CV_32FC3);
    //do we need clear input_tensor_arr_?
    this->input_tensor_arr_.clear();
    this->input_tensor_arr_.push_back(input_img);
}

bool Sku110KModelInferencer::run(){
    //Run(std::vector<tensorflow::Tensor> &inputs,std::vector<tensorflow::Tensor> &outputs)
    this->output_tensor_arr_.clear();
    bool ifsuccess = this->parser_.Run(this->input_tensor_arr_, this->output_tensor_arr_);
    if(ifsuccess ==false){
        return ifsuccess;
    }
    if (this->output_tensor_arr_.size() < 4){
        std::cerr << "output tensor arr size is:" << this->output_tensor_arr_.size() << " Expected to be 4!" << std::endl;
        return false;
    }
    this->warpOutputTensorToData();
    if (this->output_datas_.size() < 1)
        return false;
    this->filtered_datas_ = emmerger::filterDuplicateCandidates(this->output_datas_, this->img_);
    return ifsuccess;
}

void Sku110KModelInferencer::warpOutputTensorToData(){
    auto boxes_tensor = this->output_tensor_arr_[0];
    auto boxes = boxes_tensor.tensor<float,3>();
    auto hard_scores_tensor = this->output_tensor_arr_[1];
    auto hard_scores = hard_scores_tensor.tensor<float,2>();
    auto labels_tensor = this->output_tensor_arr_[2];
    auto labels = labels_tensor.tensor<int, 2>();
    auto soft_scores_tensor = this->output_tensor_arr_[3];
    auto soft_scores = soft_scores_tensor.tensor<float,3>();
    this->output_datas_.clear();
    //std::cout << "scale:" << this->scale_ << std::endl;
    auto idxInOriImg = [this](float coord)->float{
        return static_cast<float>(static_cast<int>(coord/this->scale_));
    };
    for (int i = 0; i < hard_scores.dimension(1); i++){
        float hard_score = hard_scores(0,i);
        if (hard_scores(0,i) < this->hard_score_threshold_){
            break;
        } 
        float soft_score = soft_scores(0,i,0);
        float confidence = hard_score_rate_ * hard_score + (1.0f - hard_score_rate_) * soft_score;               
        Data data(idxInOriImg(boxes(0,i,0)), idxInOriImg(boxes(0,i,1)),idxInOriImg(boxes(0,i,2)),idxInOriImg(boxes(0,i,3)), confidence, hard_score);        
        //data.print();
        this->output_datas_.push_back(data);
    }
    std::sort(this->output_datas_.begin(),this->output_datas_.end(),[](const Data& d1, const Data& d2)->bool{
        return d1.confidence > d2.confidence;});
    if (this->output_datas_.size() > 9999){
        this->output_datas_.erase(this->output_datas_.begin() + 9999, this->output_datas_.end());
    }
    // for(Data& d:this->output_datas_){
    //     d.print();
    // }    
    // std::cout << "data size:" << this->output_datas_.size() << std::endl;
}

void Sku110KModelInferencer::showBoxes(const std::string& save_fn){
    for (const auto& d:this->filtered_datas_){
        cv::rectangle(this->img_, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), cv::Scalar(0,0,255), 2);
        std::stringstream ss;
        ss << d.hard_score << ":" << d.avg_score;
        cv::putText(this->img_, ss.str(), cv::Point(d.x1, d.y1-10), cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,0,0),2);
        cv::putText(this->img_, ss.str(), cv::Point(d.x1, d.y1-10), cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1);
    }
    cv::imwrite(save_fn, this->img_);
}