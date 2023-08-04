#include "EMMerger.h"
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <iterator>
#include <opencv2/highgui.hpp>
#include "MoG.h"
#include <opencv2/dnn/dnn.hpp>

namespace emmerger{

cv::Mat roundAndInt(const cv::Mat& input_mat){
    cv::Mat round_mat = cv::Mat::zeros(input_mat.size(), CV_32S);
    for (int r = 0; r < input_mat.rows; r++){
        for(int c = 0; c < input_mat.cols; c++){
            round_mat.at<int>(r,c) = static_cast<int>(std::round(input_mat.at<float>(r,c)));
        }
    }
    return round_mat;
}

/****************** vector转Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(std::vector<_Tp> v, int channels, int rows){
	cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
	return dest;
}

Params params;
cv::Mat linespace(const int l){
    float start = static_cast<float>(-l)/2.0f;    
    float end = static_cast<float>(l)/2.0f;    
    std::vector<float> line_arr(l);
    float step = (end - start)/(l-1);    
    for (int i = 0 ; i < l; i++){        
        line_arr[i]= start + step*i;        
    }
    return convertVector2Mat(line_arr, 1, 1);
}

void meshgrid(const cv::Mat& xgv, const cv::Mat&ygv, cv::Mat1f& x_grid, cv::Mat1f& y_grid){
    cv::repeat(xgv, ygv.total(), 1, x_grid);
    cv::repeat(ygv.t(), 1, xgv.total(), y_grid);
}

cv::Mat gaussianBlur(const int w, const int h){
    float sigma_x = w/2.0f;
    float sigma_y = h/2.0f;
    cv::Mat line_x = linespace(w);   
    cv::Mat line_y = linespace(h);    
    cv::Mat1f x_grid, y_grid;
    meshgrid(line_x, line_y, x_grid, y_grid);
    const float inv_sigmax = 1.f/sqrtf(2.0f) / sigma_x;
    const float inv_sigmay = 1.f/sqrtf(2.0f) / sigma_y;
    x_grid = x_grid.mul(inv_sigmax);
    y_grid = y_grid.mul(inv_sigmay);
    cv::Mat x_grid2 = x_grid.mul(x_grid);
    cv::Mat y_grid2 = y_grid.mul(y_grid);
    cv::Mat kernel;
    cv::Mat tmp = -1.0f*(x_grid2 + y_grid2);
    cv::exp(tmp, kernel);
    return kernel;
}

cv::Mat aggregateGaussians(const std::vector<Result>& result_arr, const cv::Size& heat_map_size){
    cv::Mat heat_map(heat_map_size, CV_32F, cv::Scalar(0.0f));
    for(const auto& result:result_arr){
        cv::Mat gauss_img = gaussianBlur(result.gauss_width, result.gauss_height);        
        cv::normalize(gauss_img, gauss_img, 0.0f, result.confidence, cv::NORM_MINMAX);        
        // reshape need to be checked with size 后面再对reshape 进行判断吧 目前看来不需要做什么
        //cv::resize(gauss_img,gauss_img, result.rect.size());        
        cv::Mat roi(heat_map, result.rect);        
        roi += gauss_img;
    }
    return heat_map;
}

std::vector<DetectCeter> shrinkBoxes(const std::vector<Data>& data_arr, cv::Mat& heat_map){
    std::vector<float> width_arr(data_arr.size()), height_arr(data_arr.size());
    std::vector<DetectCeter> ori_det_centers(data_arr.size());
    std::vector<Data> boxes(data_arr.size());
    std::vector<Result> result_arr;
    for (int i =0; i < data_arr.size(); i++){
        const float &x1 = data_arr[i].x1, &x2 = data_arr[i].x2, &y1 = data_arr[i].y1, &y2 = data_arr[i].y2;
        width_arr[i] = x2 - x1;
        height_arr[i] = y2 - y1;
        boxes[i] = data_arr[i];
        int w_shift = static_cast<int>((width_arr[i]*(1-params.box_size_factor))/2.0);
        int h_shift = static_cast<int>((height_arr[i]*(1-params.box_size_factor))/2.0);
        boxes[i].x1 += w_shift;
        boxes[i].x2 -= w_shift;
        boxes[i].y1 += h_shift;
        boxes[i].y2 -= h_shift;
        float width = boxes[i].x2 - boxes[i].x1;
        float height = boxes[i].y2 - boxes[i].y1;
        const float& confidence = data_arr[i].confidence;
        ori_det_centers[i] = DetectCeter(x1 + width_arr[i]/2.0f,y1 + height_arr[i]/2.0f,boxes[i].x1,boxes[i].x2,boxes[i].y1,boxes[i].y2,width/2.0f,height/2.0f,confidence);
        result_arr.emplace_back(width,height,confidence, cv::Rect(cv::Point(boxes[i].x1, boxes[i].y1), cv::Point(boxes[i].x2, boxes[i].y2)));
    }
    heat_map = aggregateGaussians(result_arr, heat_map.size());
    return ori_det_centers;
}

std::vector<std::vector<cv::Point>> normalizeAndFindContourOfHeatMap(cv::Mat& heat_map){
    cv::normalize(heat_map, heat_map, 0,255, cv::NORM_MINMAX);
    cv::convertScaleAbs(heat_map, heat_map);
    cv::threshold(heat_map, heat_map, 4,255, cv::THRESH_TOZERO);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(heat_map,contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

std::vector<int> getContourIndex(const cv::Rect& bbox, const std::vector<DetectCeter>& ori_det_centers){
    const auto& x1 = bbox.tl().x;
    const auto& x2 = bbox.br().x;
    const auto& y1 = bbox.tl().y;
    const auto& y2 = bbox.br().y;
    std::vector<int> valid_idx;
    for(int i = 0; i < ori_det_centers.size(); i++){
        const auto& x = ori_det_centers[i].x;
        const auto& y = ori_det_centers[i].y;
        if(x1 <= x && x <= x2 && y1 <= y && y <= y2){
            valid_idx.push_back(i);
        }
    }
    return valid_idx;
}

constexpr float chi_square_val = 1.1774100225154749f;
std::tuple<std::vector<cv::Mat>,cv::Mat, int, cv::Mat> removeRedundant(const cv::Rect& bbox,const int k, const cv::Mat& image, const cv::Mat sub_heat_map, std::vector<cv::Mat>& cov, cv::Mat& mu){
    mu = roundAndInt(mu);    
    cv::Mat roi;
    image(bbox).copyTo(roi);
    std::vector<std::vector<cv::Point>> cnts;
    for (int i = 0; i < cov.size(); i++){
        const auto& c = cov[i];
        float sigmax = sqrtf(c.at<float>(0,0));
        float sigmay = sqrtf(c.at<float>(1,1));
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(c, eigenvalues, eigenvectors);        
        float angle = atan2(eigenvectors.at<float>(0,1),eigenvectors.at<float>(0,0));
        if (angle < 0){
            angle += 2*M_PI;            
        }
        angle = 180.0f * angle / M_PI;                
        float half_major_axis_size = chi_square_val * sqrtf(eigenvalues.at<float>(1));        
        float half_minor_axis_size = chi_square_val * sqrtf(eigenvalues.at<float>(0));        
        cv::Mat local_m = cv::Mat::zeros(sub_heat_map.size(), sub_heat_map.type());
        std::vector<cv::Point> poly;
        cv::ellipse2Poly(cv::Point(mu.row(i).at<int>(0),mu.row(i).at<int>(1)), \
        cv::Size(static_cast<int>(std::round(half_minor_axis_size)), static_cast<int>(std::round(half_major_axis_size))),\
        -1*static_cast<int>(std::round(angle)),0,360,15,poly);         
        const cv::Point* pts = &(poly[0]);
        int num=(int)poly.size();
        cv::fillPoly(local_m, &pts, &num, 1, cv::Scalar(255));                
        std::vector<std::vector<cv::Point>> contours;        
        cv::findContours(local_m, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        cnts.push_back(contours[0]);
    }
    cv::Mat mu_temp = mu.clone();    
    cv::Mat distances = cv::Mat::zeros(mu_temp.rows, mu_temp.rows, CV_32F);       
    for (int r = 0; r < mu_temp.rows; r++){
        for (int c = 0; c < mu_temp.rows; c++){            
            distances.at<float>(r,c) = static_cast<float>(cv::norm(mu_temp.row(r), mu_temp.row(c)));              
        }
    }    
    cv::Mat scaled_distances = cv::Mat::zeros(k,k,CV_32F);
    for (int i = 0 ; i < k; ++i){
        for(int j = i; j < k; ++j){
            if(i==j){
                scaled_distances.at<float>(i,j) = 0;
                continue;
            }
            auto cnt_i = cnts[i];
            auto cnt_j = cnts[j];
            double ct_i_to_pt_j = -1.0*cv::pointPolygonTest(cnt_i, cv::Point2f(mu.at<int>(j,0), mu.at<int>(j,1)), true);
            double ct_j_to_pt_i = -1.0*cv::pointPolygonTest(cnt_j,cv::Point2f(mu.at<int>(i,0),mu.at<int>(i,1)),true);     
            if(ct_i_to_pt_j <=0.0 || ct_j_to_pt_i <=0.0){
                scaled_distances.at<float>(i,j) = -1.0f*FLT_MAX;
            } else{
                float pt_dist = distances.at<float>(i,j);
                float ct_i_to_ct_j = ct_i_to_pt_j - pt_dist + ct_j_to_pt_i;
                scaled_distances.at<float>(i,j) = ct_i_to_ct_j;                
            }
        }
    }    
    //i_s, j_s = numpy.unravel_index(numpy.argsort(scaled_distances, axis=None), scaled_distances.shape)
    std::vector<int> i_s,j_s;
    cv::Mat scaled_distances_temp = scaled_distances.reshape(1,1);   
    cv::Mat sorted_idx;   
    cv::sortIdx(scaled_distances_temp, sorted_idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);     
    for (int i = 0; i < sorted_idx.cols; i++){
        int idx = sorted_idx.at<int>(i);
        i_s.push_back(idx/k);
        j_s.push_back(idx%k);
    }    
    std::vector<int> to_remove;
    for(int l = 0; l < i_s.size(); l++){
        int i = i_s[l], j = j_s[l];
        if(scaled_distances.at<float>(i,j) >= 0.0f)
            break;
        if(std::find(to_remove.begin(),to_remove.end(), i) == to_remove.end() &&  std::find(to_remove.begin(),to_remove.end(), j) == to_remove.end()){
            const auto& pt_i = mu_temp.row(i);
            const auto& pt_j = mu_temp.row(j);
            int pt1_x = std::max(0, std::min(pt_i.at<int>(1), sub_heat_map.rows - 1));
            int pt1_y = std::max(0, std::min(pt_i.at<int>(0), sub_heat_map.cols - 1));
            int pt2_x = std::max(0, std::min(pt_j.at<int>(1), sub_heat_map.rows - 1));
            int pt2_y = std::max(0, std::min(pt_j.at<int>(0), sub_heat_map.cols - 1));
            auto val_i = sub_heat_map.at<uchar>(pt1_x, pt1_y);
            auto val_j = sub_heat_map.at<uchar>(pt2_x, pt2_y);
            int remove_id = i;
            if(val_j < val_i){
                remove_id = j;
            }
            to_remove.push_back(remove_id);
        }
    }    

    if(to_remove.size() > 0){ 
        cv::Mat mu_keep;
        std::vector<cv::Mat> cov_keep;       
        for (int i = 0 ; i < cov.size(); ++i){
            if (std::find(to_remove.begin(),to_remove.end(), i) == to_remove.end()){
                mu_keep.push_back(mu.row(i));
                cov_keep.push_back(cov[i]);
            }
        }      
        return std::make_tuple(cov_keep, mu_keep, cov_keep.size(), roi);
    } 
    return std::make_tuple(cov, mu, cov.size(), roi);
}

void setCandidates(std::vector<Candidate>& candidates, const std::vector<cv::Mat>& cov, const cv::Mat heat_map, const cv::Mat& mu, const int num,const cv::Point& offset,const cv::Mat& sub_heat_map){
    for (int i = 0; i < cov.size(); ++i){
        float sigmax = sqrtf(cov[i].at<float>(0,0));
        float sigmay = sqrtf(cov[i].at<float>(1,1));
        int _x = mu.row(i).at<int>(0);
        int _y = mu.row(i).at<int>(1);
        int _x1 = static_cast<int>(round(std::max(0.0f, _x - 2*sigmax)));
        int _y1 = static_cast<int>(round(std::max(0.0f, _y - 2*sigmay)));
        int _x2 = static_cast<int>(round(std::min((float)sub_heat_map.cols, _x + 2*sigmax)));
        int _y2 = static_cast<int>(round(std::min((float)sub_heat_map.rows, _y + 2*sigmay)));
        cv::Rect abs_box(_x1 + offset.x, _y1+offset.y, _x2-_x1, _y2-_y1);
        if(abs_box.width > params.min_box_size && abs_box.height > params.min_box_size){
            double max_val;
            cv::minMaxLoc(heat_map(abs_box),NULL, &max_val, NULL,NULL);
            float score = static_cast<float>(max_val);
            candidates.emplace_back(abs_box, std::vector<int>{}, score);
        }
    }
}

std::vector<Candidate> findNewCandidates(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat& heat_map, 
const std::vector<Data>& data, const std::vector<DetectCeter>& original_detection_centers, const cv::Mat& image){
    std::vector<Candidate> candidates;
    for(int i = 0; i < contours.size(); i++){
        auto contour_bounding_rect = cv::boundingRect(contours[i]);
        int box_width = contour_bounding_rect.width;
        int box_height = contour_bounding_rect.height;
        double contour_area = cv::contourArea(contours[i]);
        cv::Point offset = contour_bounding_rect.tl();
        auto original_indexes = getContourIndex(contour_bounding_rect, original_detection_centers);
        int n = original_indexes.size();
        if(n > 0 && box_width > 3 && box_height > 3){
            std::vector<Data> curr_data;
            std::vector<DetectCeter> curr_ori_det_centers;
            for(const int& idx : original_indexes){
                curr_data.push_back(data[idx]);
                curr_ori_det_centers.push_back(original_detection_centers[idx]);
            }
            std::vector<float> areas;
            for(const auto& d :curr_data){
                float w = (d.x2 - d.x1)*params.box_size_factor;
                float h = (d.y2 - d.y1)*params.box_size_factor;
                areas.push_back(w*h);
            }
            const auto median_it = areas.begin() + areas.size() / 2;
            std::nth_element(areas.begin(), median_it , areas.end());
            auto median_area = *median_it;
            int approximate_number_of_objects = 0;
            if(median_area > 0){
                approximate_number_of_objects = std::min(static_cast<int>(std::round(contour_area/median_area)),100);
            }
            cv::Mat sub_heat_map = heat_map(contour_bounding_rect).clone();
            int k = std::max(1, int(approximate_number_of_objects));
            if(n > k){
                if(k > params.min_k){
                    //return std::make_tuple(beta,mu_prime,covariance_prime);
                    auto collapse_result = collapse(curr_ori_det_centers, k, offset);
                    auto beta = std::get<0>(collapse_result);
                    auto mu_prime = std::get<1>(collapse_result);
                    auto cov_prime = std::get<2>(collapse_result);
                    auto rm_redundant_result = removeRedundant(contour_bounding_rect, k, image, sub_heat_map, cov_prime, mu_prime);
                    //return std::make_tuple(cov, mu, cov.size(), roi);
                    auto cov = std::get<0>(rm_redundant_result);
                    auto mu = std::get<1>(rm_redundant_result);
                    setCandidates(candidates, cov, heat_map, mu, cov.size(), offset, sub_heat_map);
                }
            }
        }
    }
    return candidates;
}

std::vector<Candidate> mapOriginalBoxesToNewBoxes(std::vector<Candidate>& candidates, const std::vector<DetectCeter>& ori_det_centers){
    cv::Mat x(1, ori_det_centers.size(), CV_32F);
    cv::Mat y(1, ori_det_centers.size(), CV_32F);
    cv::Mat matched_indexes(1, ori_det_centers.size(), CV_8U, cv::Scalar(0));
    for (int i = 0; i < ori_det_centers.size(); i++){
        x.at<float>(0,i) = static_cast<float>(ori_det_centers[i].x);
        y.at<float>(0,i) = static_cast<float>(ori_det_centers[i].y);        
    }    
    for(auto& candidate : candidates){
        const auto& box = candidate.box;
        cv::Mat original_indexes = (box.tl().x <= x) & (x <= box.br().x) & (box.tl().y <= y) & (y <= box.br().y) & (1-matched_indexes);
        matched_indexes.setTo(1, original_indexes);
        std::vector<int> ids;
        for(int i = 0; i < original_indexes.cols; i++){
            if(original_indexes.at<uchar>(i) > 0){
                ids.push_back(i);
            }
        }
        candidate.originalDetectionIds.clear();
        std::copy(ids.begin(), ids.end(), std::back_inserter(candidate.originalDetectionIds));
    }
    std::vector<Candidate> new_candidates;
    for(const auto& c : candidates){
        if(c.originalDetectionIds.size() > 0){
            new_candidates.push_back(c);
        }
    }
    // std::cout << "new_candiates size:" << new_candidates.size() <<  std::endl;
    // for(auto& c:new_candidates){
    //     c.print();
    // }
    return new_candidates;
}

std::vector<DataWithAvgScore> performNmsOnImageDataframe(const std::vector<DataWithAvgScore>& filtered_data, const float nms_score);

std::vector<DataWithAvgScore> filterDuplicateCandidates(const std::vector<Data>& data, const cv::Mat& image){
    cv::Mat heat_map = cv::Mat::zeros(image.rows, image.cols, CV_32F);
    auto original_detection_centers = shrinkBoxes(data, heat_map);
    auto contours = normalizeAndFindContourOfHeatMap(heat_map);
    std::cout << "normalizeAndFindContourOfHeatMap finished!" << std::endl;
    auto candidates = findNewCandidates(contours, heat_map, data, original_detection_centers, image);
    std::cout << "findNewCandidates finished!" << std::endl;
    candidates = mapOriginalBoxesToNewBoxes(candidates, original_detection_centers);
    std::cout << "mapOriginalBoxesToNewBoxes finished!" << std::endl;
    std::vector<DataWithAvgScore> filtered_data;
    for(int i = 0; i < candidates.size(); i++){
        const auto& label = candidates[i].originalDetectionIds;        
        DataWithAvgScore best_detection;
        float max_avg_score = 0.0;
        for(const auto& l:label){
            float avg_score = 0.5 * data[l].confidence + 0.5 * data[l].hard_score;
            if(avg_score > max_avg_score){
                max_avg_score = avg_score;
                best_detection = DataWithAvgScore(data[l], avg_score);
            }
        }
        // std::cout << "best_detection:" << std::endl;
        // best_detection.print();
        filtered_data.push_back(best_detection);
    }   
    auto final_filtered_data = performNmsOnImageDataframe(filtered_data, 0.3f); 
    return final_filtered_data;
}

std::vector<DataWithAvgScore> performNmsOnImageDataframe(const std::vector<DataWithAvgScore>& filtered_data, const float nms_score){
    std::vector<cv::Rect> rect_arr;
    std::vector<float> confidence_arr;
    for(const auto& data : filtered_data){
        rect_arr.emplace_back(data.x1, data.y1, data.x2-data.x1, data.y2-data.y1);        
        confidence_arr.push_back(data.confidence);
    }
    // std::cout << "confidence_arr:" << std::endl;
    // std::copy(confidence_arr.begin(), confidence_arr.end(), std::ostream_iterator<float>(std::cout," "));
    //std::cout << std::endl;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rect_arr, confidence_arr, 0.6f, 0.3f, indices);
    // std::cout << "indices:" << std::endl;
    //std::copy(indices.begin(), indices.end(), std::ostream_iterator<int>(std::cout , " "));
    std::cout << std::endl;
    std::vector<DataWithAvgScore> final_fitered_data;    
    for(const auto& idx:indices){        
        final_fitered_data.push_back(filtered_data[idx]);
    }
    std::cout << "final filtered data:" << std::endl;
    for(auto& d:final_fitered_data){
        d.print();
    }
    return final_fitered_data;
}

}