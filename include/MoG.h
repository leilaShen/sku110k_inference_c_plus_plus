#include "Sku110KStructures.h"
namespace emmerger{
    
std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> agglomerativeInit(const cv::Mat& alpha, const cv::Mat& mu, const std::vector<cv::Mat>& covariance, const int n, const int k);
std::pair<float,int> minKl(const cv::Mat& beta, const cv::Mat& cov, const std::vector<cv::Mat>& covariance_prime, const cv::Mat& mu, const cv::Mat& mu_prime);

std::pair<CDict, InvDict> eStep(const cv::Mat& alpha, const cv::Mat& beta, const std::vector<cv::Mat>& covariance, const std::vector<cv::Mat>& covariance_prime,
const cv::Mat& mu, const cv::Mat& mu_prime, CacheDict& min_kl_cache);

void mStep(const cv::Mat& alpha, const CDict& clusters, const std::vector<cv::Mat>& covariance,const cv::Mat& mu,
cv::Mat& beta,std::vector<cv::Mat>& covariance_prime,  cv::Mat& mu_prime);

std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> collapse(const std::vector<DetectCeter>& ori_det_centers, const int k, const cv::Point& offset, int max_iter = 100, float epsilon = 1e-100);

}