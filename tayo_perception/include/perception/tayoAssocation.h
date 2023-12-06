#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "perception/KalmanFilter.h"
#include "perception/tayoKalmanFilter.h"

#define MAXN 1000
#define INF 1e9;
#define MAXLIFETIME 6
#define MAXCOST 255.0
static int MAX_EXTRACT = 4;

static int TrackID = 0;
static double m_time;

float euclideanDist(cv::Point2f& p, cv::Point2f& q);
class Track
{
public:
    
    double dt;
    ros::Time prev_time;
    int n = 4; // Number of states
    int m = 2; // Number of measurements
    int c = 2; // Number of control inputs

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n); // System dynamics matrix
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, c); // Input control matrix
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, n); // Output matrix
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n); // Process noise covariance
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(m, m); // Measurement noise covariance
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n); // Estimate error covariance

    KalmanFilter kf;

    int ID;
    int life_time;

    cv::Point2f cur_pos;
    std::vector<cv::Point2f> prev_pos;
    cv::Point2f predict_pos;

    cv::Point2f z_candi; // 측정치
    cv::Point2f fore_candi; // 추정치 이거로 그림
    cv::Point2f predict_candi; // 예측치

    std::vector<cv::Point2f> velocity_arr;
    std::vector<float> speed_arr;

    Track();
    ~Track();
    void tracking(cv::Point2f obj, double dt);
    void unmatchedTrack();
    void setupKalman();
    void tayoSetupKalman();
    cv::Point2f get_prev_pos()
    {
        return prev_pos[prev_pos.size()-2];
    }
};

class HungarianAlgorithm
{
public:
    int n_size, Match_num;                            // worker 수
    float label_x[MAXN], label_y[MAXN];          // label x, y
   int yMatch[MAXN];                           // y와 match되는 x
    int xMatch[MAXN];                           // x와 match되는 y
    bool S[MAXN], T[MAXN];                      // 알고리즘 상에 포함되는 vertex.
    float slack[MAXN];
    float slackx[MAXN];
    int parent[MAXN];                           // alternating path
    float cost[MAXN][MAXN];                       // cost
    float init_cost[MAXN][MAXN];                       // 초기 cost

    void init_labels();
    void update_labels();
    void add_to_tree(int x, int parent_x);
    void augment();
    void hungarian();

    void AssociaTion(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float DIST_TH, double dt);
    void trackCostFunction(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float dist_th);
    void measureCostFunction(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float dist_th);

    HungarianAlgorithm();
    ~HungarianAlgorithm();
};

class tayoAssociation
{
public:
    std::vector<Track> m_track;
    
    tayoAssociation(){}
    ~tayoAssociation(){}

    void TrackManagement(pcl::PointCloud<pcl::PointXYZI> &measure, ros::Duration run_time);
    std::vector<cv::Point2f> pointcloudTovector(pcl::PointCloud<pcl::PointXYZI> pt);
};
