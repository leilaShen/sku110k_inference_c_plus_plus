#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <iostream>
#include <iterator>
#include <map>

namespace emmerger{

struct Data{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    float hard_score;
    std::string uuid = "object_label";
    std::string label_type = "object_label";
    std::string image_name = "image_fn";
    Data() = default;    
    Data(const float x1_, const float y1_, const float x2_, const float y2_, const float& conf, const float& hs):x1(x1_), y1(y1_), x2(x2_), y2(y2_),confidence(conf), hard_score(hs){};
    Data(const float x1_, const float y1_, const float x2_, const float y2_, const float& conf, const float& hs,
    const std::string& u, const std::string& l, const std::string& img_name):x1(x1_), y1(y1_), x2(x2_), y2(y2_),confidence(conf), hard_score(hs),
    uuid(u), label_type(l), image_name(img_name){};
    Data(const Data& d){
        x1= d.x1; x2=d.x2; y1=d.y1; y2=d.y2; confidence = d.confidence;
        hard_score = d.hard_score;
        uuid = d.uuid;
        label_type = d.label_type;
        image_name = d.image_name;
    }
    void print(){
        std::cout << x1 << "," << y1 << "," << x2 << "," << y2 << "," << confidence << "," << hard_score  << std::endl;
     }
};

struct DetectCeter{
    float x;
    float y;
    float left_x;
    float right_x;
    float top_y;
    float bottom_y;
    float sigma_x;
    float sigma_y;
    float confidence;
    DetectCeter() = default;
    DetectCeter(const DetectCeter& dc):x(dc.x),y(dc.y),left_x(dc.left_x), right_x(dc.right_x),
    top_y(dc.top_y), bottom_y(dc.bottom_y), sigma_x(dc.sigma_x), sigma_y(dc.sigma_y), confidence(dc.confidence){};
    DetectCeter(const float x_, const float y_, const float lx,const float rx, const float ty, 
    const float by, const float sx, const float sy, const float c):x(x_),y(y_),left_x(lx), right_x(rx),
    top_y(ty),bottom_y(by), sigma_x(sx), sigma_y(sy), confidence(c){};
};

struct Result{
    int gauss_width;
    int gauss_height;
    float confidence = 0.0f;
    cv::Rect rect;
    Result(const int w, const int h, const float c, const cv::Rect& r):gauss_width(w),
    gauss_height(h), confidence(c), rect(r){};
};

struct Params{
    float box_size_factor = 0.5;
    float min_box_size = 5;
    float ellipsoid_thresh = 0.5;
    int min_k = 0;
    Params()=default;
    Params(const float bsf, const float mbs, const float eth, const int k):box_size_factor(bsf), 
    min_box_size(mbs), ellipsoid_thresh(eth), min_k(k){};
};

struct DataWithAvgScore{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    float hard_score;
    float avg_score;
    std::string uuid;
    std::string label_type;
    std::string image_name;
    DataWithAvgScore() = default;    
    DataWithAvgScore(const float& x1_, const float& y1_, const float& x2_, const float& y2_, const float& conf, const float& hs, const float& as):
    x1(x1_), y1(y1_), x2(x2_), y2(y2_),confidence(conf), hard_score(hs), avg_score(as){};
    DataWithAvgScore(const DataWithAvgScore& d){
        x1= d.x1; x2=d.x2; y1=d.y1; y2=d.y2; confidence = d.confidence;
        hard_score = d.hard_score;
        avg_score = d.avg_score;
        uuid = d.uuid;
        label_type = d.label_type;
        image_name = d.image_name;
    }
    DataWithAvgScore(const Data& d, const float avg_s){
        x1= d.x1; x2=d.x2; y1=d.y1; y2=d.y2; confidence = d.confidence;
        hard_score = d.hard_score;        
        uuid = d.uuid;
        label_type = d.label_type;
        image_name = d.image_name;
        avg_score = avg_s;
    }
    DataWithAvgScore(const Data& d){
        x1= d.x1; x2=d.x2; y1=d.y1; y2=d.y2; confidence = d.confidence;
        hard_score = d.hard_score;        
        uuid = d.uuid;
        label_type = d.label_type;
        image_name = d.image_name;        
    }
    void print(){
        std::cout << "x1:" << x1 << ",y1:" << y1 << ",x2:" << x2 << ",y2:" << y2 << 
        ",confidence:" << confidence << ",hard_score:" << hard_score << ",avg_score:" << avg_score << std::endl;
    }
};

struct Candidate{
    cv::Rect box;
    std::vector<int> originalDetectionIds;
    float score;
    Candidate(const cv::Rect& b, const std::vector<int>& ids, const float s){
        box = b;
        std::copy(ids.begin(), ids.end(),std::back_inserter(originalDetectionIds));
        score = s;
    }
    Candidate() = default;
    void print(){
        std::cout << "box:" << box << std::endl;
        std::cout << "original detections ids:";
        std::copy(originalDetectionIds.begin(), originalDetectionIds.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        std::cout << "score:" << score << std::endl;
    }
};

//cdict means cluster dict, corresponding to python's dict
using CDict = std::map<int,std::vector<int>>;
using InvDict = std::map<int, int>;
using CacheDict = std::map<int, std::pair<float,int>>;

}