#include "TensorflowModelParser.h"
#include "Sku110KModelInferencer.h"
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;



int main(int argc, char* argv[]){
    std::string model_path = "/media/vision/Data/work/keras_to_tensorflow/model/retinanet_emmerger.pb";
    Sku110KModelInferencer inferencer(model_path);
    inferencer.init();
    cv::Mat img = cv::imread("/media/vision/Data/work/sku_110k/SKU110K_CVPR19/WHYY/images/181_1_182499_4103175824_100012816042_RGB_2020-09-23-14-34-07-999.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    std::cout << "start-------------------------" << std::endl;
    auto start_time = Clock::now();
    inferencer.preprocess(img);    
    bool ifsuccess = inferencer.run();
    auto end_time = Clock::now();
	std::cout << "Time difference:"
      << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
    inferencer.showBoxes("181.png");
    //second image
    img = cv::imread("/media/vision/Data/work/sku_110k/SKU110K_CVPR19/WHYY/images/185_1_182499_4103175824_100010617232_RGB_2020-09-23-14-45-13-999.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    std::cout << "second image -------------------------" << std::endl;
    start_time = Clock::now();
    inferencer.preprocess(img);    
    ifsuccess = inferencer.run();
    end_time = Clock::now();
	std::cout << "Time difference:"
      << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
    inferencer.showBoxes("185.png");
    return 0;
}