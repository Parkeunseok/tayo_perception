#include <iostream>
#include <ros/ros.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <geometry_msgs/Point.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <tf2/transform_datatypes.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <visualization_msgs/MarkerArray.h>
#include <obstacle_msgs/obstacleArray.h>
// #include <decision_msgs/decision

#include <decision_msgs/decisionData.h>
#include "perception/tayoAssocation.h"

namespace Perception
{
    class tayoPerception
    {
    private:
        ros::NodeHandle nh_;

        // PUB & SUB Topic name
        std::string hesai_topic_;
        std::string sub_lidar_topic_;
        std::string sub_status_topic_;
        std::string hesai_object_pcl_topic_;
        std::string hesai_globalfilter_pcl_topic_;
        std::string hesai_object_pt_topic_;
        std::string hesai_object_polygon_topic_;
        std::string hesai_object_info_topic_;
        std::string map_frame_id_;
        std::string lidar_frame_id_;
        std::string sub_lidar_frame_id_;

        // ROS Subscriber & Publisher
        ros::Subscriber sub_hesai_;
        ros::Subscriber sub_livox_;
        ros::Subscriber sub_status_;
        ros::Publisher pub_object_info_;
        ros::Publisher pub_object_pcl_;
        ros::Publisher pub_mapfilter_pcl_;
        ros::Publisher pub_object_pt_;
        ros::Publisher pub_sublidar_pcl_;
        ros::Publisher pub_polygon_;
        ros::Publisher pub_predict;

        // hesai filter & object data
        pcl::PointCloud<pcl::PointXYZI> hesai_globalfilter_pcl_;
        pcl::PointCloud<pcl::PointXYZI> hesai_object_pcl_;

        // hesai object point#include "perception/tayoAssociation.h"XYZI> hesai_object_pt_;
        pcl::PointCloud<pcl::PointXYZI> main_lidar_raw_;
        pcl::PointCloud<pcl::PointXYZI> sub_lidar_raw_;

        std::string mode_;

        // map info
        std::string map_path_;
        std::string stop_1_map_path_, stop_4_map_path_;
        std::string map_;
        int map_flag_;
        cv::Mat map_info_;
        cv::Mat stop_1_map_, stop_4_map_;
        geometry_msgs::Point init_pose_;

        decision_msgs::decisionData status_;

        // car state
        int cur_lab_;
        int final_lab_;
        int rank_;

        // ROI Filter param
        geometry_msgs::Point max_distance_;
        geometry_msgs::Point min_distance_;

        // map filter param
        double min_thresh_;
        double min_long_thresh_;
        double max_thresh_;
        double min_dist_, max_dist_;
        double cur_thresh_;

        // ground Remover param
        int grid_dim_;
        double per_cell_;
        double height_threshold_;
        double close_pt_min_, close_pt_max_;

        int remove_size_;

        // euclidean cluster param
        double range_cluster_;
        int min_cluster_size_;
        int max_cluster_size_;

        // sub lidar transformation
        double sub_lidar_pose_x_;
        double sub_lidar_pose_y_;
        double sub_lidar_pose_z_;
        double sub_lidar_ori_x_;
        double sub_lidar_ori_y_;
        double sub_lidar_ori_z_;
        double sub_lidar_ori_w_;

        // flag
        bool is_lidar_;
        bool is_hesai_;
        bool is_velodyne_;
        bool is_tf_;

        // sensor info
        ros::Time lidar_time_;

        // transform
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;


        // Tracking
        tayoAssociation track;
        double dt;
        ros::Time prev_time;
        ros::Duration time;
        int MAX_EXTRACT = 4;

        visualization_msgs::MarkerArray m_speed_result;
        ros::Publisher id_pub;
    public:
        tayoPerception(ros::NodeHandle &nh);
        ~tayoPerception();

        void initSensorInfo();
        void initPerceptionParam();

        void callbackHesai(const sensor_msgs::PointCloud2Ptr msg);
        void callbackSublidar(const sensor_msgs::PointCloud2Ptr msg);
        void callbackSubstatus(const decision_msgs::decisionData::ConstPtr &msg);

        void run(geometry_msgs::TransformStamped transformStamped,
                 geometry_msgs::TransformStamped inv_transformStamped,
                 ros::Publisher global_filtered_pub,
                 ros::Publisher ground_filtered_pub,
                 ros::Publisher object_pt_pub,
                 std::string frame_id,
                 pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                 double dt);
        void pubObject(ros::Publisher pub,
                       pcl::PointCloud<pcl::PointXYZI> cloud_in,
                       std::string frame_id);
        void pubObjectInfo(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZI> obj_cluster, std::vector<pcl::PointCloud<pcl::PointXYZI>> point_cloud, std::string frame_id);
        void makePolygon(pcl::PointCloud<pcl::PointXYZI> point_cloud, int id, visualization_msgs::Marker &polygon);
        void makeObjPt(pcl::PointCloud<pcl::PointXYZI> point_cloud, pcl::PointXYZI &minPt);

        void noiseRemover(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI> &cloud_out);
        void groundRemover(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                           pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                           pcl::PointCloud<pcl::PointXYZI> &cloud_grid_out);
        cv::Mat selectMap();
        void mapFilter(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                        pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                        pcl::PointCloud<pcl::PointXYZI> &cloud_map,
                        geometry_msgs::Pose pose,
                        bool is_over);
        void roiFilter(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                       pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                       pcl::PointCloud<pcl::PointXYZI> &cloud_over);
        void roiVelodyne(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                         pcl::PointCloud<pcl::PointXYZI> &cloud_out);
        void extractMinPt(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                          std::vector<pcl::PointCloud<pcl::PointXYZI>> &cluster_pc);
        void extractCenPt(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI> &cloud_out);
        void extractMeanPt(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI> &cloud_out);
        void matchCluster(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI> &cloud_out);
        void publish_polygon(std::vector<pcl::PointCloud<pcl::PointXYZI>> hull);
        
        void makeID();
      
        
    };
}