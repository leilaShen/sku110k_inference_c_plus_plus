#include "MoG.h"
#include <iostream>
#include <map>

namespace emmerger {

std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> agglomerativeInit(const cv::Mat& alpha, const cv::Mat& mu, const std::vector<cv::Mat>& covariance, const int n, const int k){
    cv::Mat mu_copy = mu.clone();
    cv::Mat mu_stack = cv::Mat::zeros(n-k,2,mu.type()) + FLT_MAX;
    cv::Mat mu_temp;
    cv::vconcat(mu_copy, mu_stack, mu_temp);
    //std::cout << "mu_temp:" << mu_temp << std::endl;
    
    cv::Mat alpha_temp;
    cv::vconcat(alpha, cv::Mat::zeros((n-k),1,alpha.type()), alpha_temp);
    //std::cout << "alpha_temp:" << alpha_temp << std::endl;
    std::vector<cv::Mat> cov_stack(n-k,cv::Mat::zeros(2,2, CV_32F));
    std::vector<cv::Mat> covariance_temp(covariance);
    covariance_temp.insert(covariance_temp.end(), cov_stack.begin(),cov_stack.end());
    
    cv::Mat distances = cv::Mat::zeros(mu_temp.rows, mu_temp.rows, mu_temp.type())+FLT_MAX;       
    for (int r = 0; r < mu_temp.rows; r++){
        for (int c = r; c < mu_temp.rows; c++){            
            distances.at<float>(r,c) = cv::norm(mu_temp.row(r), mu_temp.row(c));            
        }
    }
    //std::cout << "distances:" << distances << std::endl;
    distances.setTo(FLT_MAX, distances == 0.0f);
    //std::cout << "distances:\n" << distances << std::endl;
    std::vector<int> deleted;
    for(int l = n; l < 2*n-k; l++){
        cv::Point min_loc;
        cv::minMaxLoc(distances, NULL, NULL, &min_loc, NULL);
        int i = min_loc.y, j = min_loc.x;
        float alpha_i = alpha_temp.at<float>(i,0);
        float alpha_j = alpha_temp.at<float>(j,0);
        float alpha_ij = alpha_i + alpha_j;
        cv::Mat mu_ij = (alpha_i * mu_temp.row(i) + alpha_j * mu_temp.row(j))/alpha_ij;
        float harmonic_mean = (alpha_i * alpha_j)/alpha_ij;
        cv::Mat delta_mu = (mu_temp.row(i) - mu_temp.row(j)).t();
        cv::Mat covariance_ij = (alpha_i * covariance_temp[i] + alpha_j * covariance_temp[j]
        + harmonic_mean * (delta_mu*delta_mu.t())) / alpha_ij;
        mu_ij.copyTo(mu_temp.row(l));
        covariance_ij.copyTo(covariance_temp[l]);
        alpha_temp.at<float>(l,0) = alpha_ij;
        distances.col(i).setTo(FLT_MAX);
        distances.col(j).setTo(FLT_MAX);
        distances.row(i).setTo(FLT_MAX);
        distances.row(j).setTo(FLT_MAX);
        mu_temp.row(i).setTo(FLT_MAX);
        mu_temp.row(j).setTo(FLT_MAX);
        deleted.push_back(i);
        deleted.push_back(j);
        //std::cout << "deleted:" << i << "\t" << j << std::endl;
        cv::Mat d = cv::Mat::zeros(mu_temp.rows,1, mu_temp.type());
        for (int k = 0; k < mu_temp.rows; k++){
            d.at<float>(k) = cv::norm(mu_temp.row(k), mu_ij);
        }
        d.setTo(FLT_MAX, d==0.0f);
        d.copyTo(distances.col(l));
    }
    
    cv::Mat alpha_keep, mu_keep;
    std::vector<cv::Mat> covariance_keep;
    for (int i = 0; i < alpha_temp.rows; i++){
        if(std::find(deleted.begin(), deleted.end(), i) != deleted.end()){
            continue;
        }
        alpha_keep.push_back(alpha_temp.row(i));
        mu_keep.push_back(mu_temp.row(i));
        covariance_keep.push_back(covariance_temp[i]);
    }
    return std::make_tuple(alpha_keep, mu_keep, covariance_keep);
}

std::pair<float,int> minKl(const cv::Mat& beta, const cv::Mat& cov, const std::vector<cv::Mat>& covariance_prime, const cv::Mat& mu, const cv::Mat& mu_prime){
    cv::Mat cov_g = cv::Mat::zeros(mu_prime.size(), CV_32F);
    cv::Mat sigma_xx, sigma_yy;
    for(int i = 0; i < covariance_prime.size(); i++){
        sigma_xx.push_back(covariance_prime[i].at<float>(0,0));
        sigma_yy.push_back(covariance_prime[i].at<float>(1,1));
    }
    sigma_xx = sigma_xx.reshape(1,covariance_prime.size());
    sigma_yy = sigma_yy.reshape(1,covariance_prime.size());
    sigma_xx.copyTo(cov_g.col(0));
    sigma_yy.copyTo(cov_g.col(1));
    cv::Mat cov_f = cv::Mat::zeros(mu_prime.size(), CV_32F);
    cov_f.col(0).setTo(cov.at<float>(0,0));
    cov_f.col(1).setTo(cov.at<float>(1,1));
    cv::Mat mu_f = cv::Mat::zeros(mu_prime.size(), CV_32F);
    mu_f.col(0).setTo(mu.at<float>(0));
    mu_f.col(1).setTo(mu.at<float>(1));
    cv::Mat mu_g = mu_prime.clone();
    cv::Mat cov_g_sqrt, cov_f_sqrt;
    cv::sqrt(cov_g, cov_g_sqrt);
    cv::sqrt(cov_f, cov_f_sqrt);
    cv::Mat log_ratio_0, log_ratio_1;
    cv::log(cov_g_sqrt.col(0)/cov_f_sqrt.col(0), log_ratio_0);
    cv::log(cov_g_sqrt.col(1)/cov_f_sqrt.col(1), log_ratio_1);
    cv::Mat log_ratio = log_ratio_0 + log_ratio_1; 
    //std::cout << "log_ratio:" << log_ratio << std::endl;
    cv::Mat delta_mu = mu_f - mu_g;
    cv::Mat delta_mu_square = delta_mu.mul(delta_mu);
    cv::Mat div = (cov_f.col(0) + delta_mu_square.col(0))/(2*cov_g.col(0)) + (cov_f.col(1) + delta_mu_square.col(1))/(2*cov_g.col(1));
    cv::Mat kl = div + log_ratio;
    //std::cout << "kl:" << kl << std::endl;
    double min_val;
    cv::Point min_idx;
    cv::minMaxLoc(kl, &min_val, NULL, &min_idx, NULL);
    //std::cout << "min_val:" << min_val << "\tmin_idx:" << min_idx.y << std::endl;
    return std::make_pair(static_cast<float>(min_val), min_idx.y);
}

std::pair<CDict, InvDict> eStep(const cv::Mat& alpha, const cv::Mat& beta, const std::vector<cv::Mat>& covariance, const std::vector<cv::Mat>& covariance_prime,
const cv::Mat& mu, const cv::Mat& mu_prime, CacheDict& min_kl_cache){
    CDict clusters;
    InvDict clusters_inv;
    for(int i = 0; i < covariance.size(); i++){
        float min_dist = 0.0;
        int selected_cluster = 0;
        if(min_kl_cache.find(i) != min_kl_cache.end()){
            auto info = min_kl_cache[i];
            min_dist = info.first;
            selected_cluster = info.second;
        } else {
            auto info = minKl(beta,covariance[i], covariance_prime, mu.row(i), mu_prime);
            min_dist = info.first;
            selected_cluster = info.second;
        }
        //std::cout << "min_dist:" << min_dist << "\tselected_cluster:" << selected_cluster << std::endl;
        if(clusters.find(selected_cluster) == clusters.end()){
            std::vector<int> cluster_arr;
            clusters[selected_cluster] = cluster_arr;
        }
        clusters[selected_cluster].push_back(i);
        clusters_inv[i] = selected_cluster; 
    }
    return std::make_pair(clusters, clusters_inv);
}

void mStep(const cv::Mat& alpha, const CDict& clusters, const std::vector<cv::Mat>& covariance,const cv::Mat& mu,
cv::Mat& beta,std::vector<cv::Mat>& covariance_prime,  cv::Mat& mu_prime){
    for(auto iter = clusters.begin(); iter != clusters.end(); ++iter){
        int j = iter->first;
        //std::cout << j << std::endl;
        const auto& t_vals = iter->second;
        float beta_update = 0.0;
        cv::Mat mu_update = cv::Mat::zeros(1,2,CV_32F);        
        for(const auto& t:t_vals){
            //std::cout << "t:" << t << std::endl;
            beta_update += alpha.at<float>(t);
            mu_update = mu_update + alpha.at<float>(t)*mu.row(t);
        }
        //std::cout << "beta_update:" << beta_update << std::endl;
        beta.at<float>(j) = beta_update;
        mu_update = mu_update/beta_update;
        //std::cout << "mu_update:" << mu_update << std::endl;
        mu_update.copyTo(mu_prime.row(j));
        //update covariance
        cv::Mat cov_update = cv::Mat::zeros(2,2,CV_32F);        
        //std::cout << "mu:" << mu << std::endl;
        for (const auto& t:t_vals){            
            cv::Mat delta_mu = mu.row(t) - mu_prime.row(j);
            cov_update = cov_update + alpha.at<float>(t)*(covariance[t] + delta_mu.t()*delta_mu);            
        }
        cov_update = cov_update/beta.at<float>(j);
        cov_update.copyTo(covariance_prime[j]);
    }
}

std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> collapse(const std::vector<DetectCeter>& ori_det_centers, const int k, const cv::Point& offset, int max_iter, float epsilon){
    // assemble alpha, mu, covariance first
    int n = ori_det_centers.size();
    //std::cout << "n:" << n << std::endl;
    cv::Mat mu_x, mu_y, alpha;
    float sigma_xx, sigma_yy;
    std::vector<cv::Mat> covariance;
    float confidence_sum = 0.0;
    for (int i = 0 ; i < n; i++){
        mu_x.push_back(ori_det_centers[i].x - offset.x);
        mu_y.push_back(ori_det_centers[i].y - offset.y);
        sigma_xx = ori_det_centers[i].sigma_x * ori_det_centers[i].sigma_x;
        sigma_yy = ori_det_centers[i].sigma_y * ori_det_centers[i].sigma_y;
        //std::cout << "confidence:" << ori_det_centers[i].confidence << std::endl;
        alpha.push_back(ori_det_centers[i].confidence);
        confidence_sum += ori_det_centers[i].confidence;
        covariance.push_back((cv::Mat_<float>(2,2) << sigma_xx,0.0,0.0,sigma_yy));
    }
    // std::cout << "mu_x:" << mu_x << std::endl;
    // std::cout << "mu_y:" << mu_y << std::endl;
    // std::cout << "alpha:" << alpha << std::endl;
    alpha = alpha.reshape(1,n) / confidence_sum;
    //std::cout << "alpha:" << alpha << std::endl;
    mu_x = mu_x.reshape(1,n);
    mu_y = mu_y.reshape(1,n);
    cv::Mat mu;
    cv::hconcat(mu_x, mu_y, mu);
    //std::cout << "mu:" << mu << std::endl;
    // init clusters
    auto init_result = agglomerativeInit(alpha, mu, covariance, n, k);
    auto beta = std::get<0>(init_result);
    auto mu_prime = std::get<1>(init_result);
    auto covariance_prime = std::get<2>(init_result);
    //try EM
    //first, save the init result, if EM failed, reture the init result
    cv::Mat beta_init = beta.clone();
    cv::Mat mu_prime_init = mu_prime.clone();
    std::vector<cv::Mat> covariance_prime_init;
    for(const auto& c : covariance_prime){
        covariance_prime_init.push_back(c.clone());
    }
    // begin...
    int iteration = 0;
    float d_val = FLT_MAX;
    float delta = FLT_MAX;
    float prev_d_val = FLT_MAX;
    CacheDict min_kl_cache;
    while(delta > epsilon && iteration < max_iter){
        iteration++;
        auto cluster_info = eStep(alpha, beta, covariance, covariance_prime, mu, mu_prime, min_kl_cache);
        auto clusters = cluster_info.first;
        auto clusters_inv = cluster_info.second;
        mStep(alpha,clusters,covariance,mu,beta,covariance_prime,mu_prime);
        prev_d_val = d_val;
        d_val = 0.0;
        for(int t = 0; t < covariance.size(); t++){
            float alpha_ = alpha.at<float>(t);
            auto seleted_cluster_info = minKl(beta, covariance[t], covariance_prime, mu.row(t), mu_prime);
            float min_dist = seleted_cluster_info.first;
            int selected_cluster = seleted_cluster_info.second;
            d_val += alpha_ * min_dist;
        }
        delta = prev_d_val - d_val;
        if (delta < 0){
            std::cerr << "EM bug - not monotonic - using fallback" << std::endl;
            return std::make_tuple(beta_init, mu_prime_init, covariance_prime_init);
        }
    }
    if(delta > epsilon){
        std::cerr << "EM did not converge - using fallback" << std::endl;
        return std::make_tuple(beta_init, mu_prime_init, covariance_prime_init);
    }
    return std::make_tuple(beta,mu_prime,covariance_prime);
}

}
