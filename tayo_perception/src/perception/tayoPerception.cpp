#include "perception/tayoPerception.h"

namespace Perception
{
    tayoPerception::tayoPerception(ros::NodeHandle &nh)
                                   : nh_(nh), tf_buffer_(), tf_listener_(tf_buffer_), track()
    {
        // initialize perception param
        initSensorInfo();
        initPerceptionParam();

        // check load map data
        if(map_ == "speedway"){
            map_info_ = cv::imread(map_path_, cv::IMREAD_COLOR);
            stop_1_map_ = cv::imread(stop_1_map_path_, cv::IMREAD_COLOR);
            stop_4_map_ = cv::imread(stop_4_map_path_, cv::IMREAD_COLOR);
            if (!map_info_.empty() && !stop_1_map_.empty() && !stop_4_map_.empty())
            {
                ROS_INFO("Success to all map data");
            }
            else
            {
                if(map_info_.empty()){
                    ROS_ERROR("Fail to main map data");
                    return;
                }
                if(map_info_.empty()){
                    ROS_ERROR("Fail to stop 1-3 map data");
                    return;
                }
                if(map_info_.empty()){
                    ROS_ERROR("Fail to stop 4-6 map data");
                    return;
                }
            }
        }   
        else{
            map_info_ = cv::imread(map_path_, cv::IMREAD_COLOR);
            if (map_info_.empty())
            {
                ROS_ERROR("Fail to map data");
                return;
            }
            else
            {
                ROS_INFO("Success to map data");
            }
        }
        std::cout << "map flag " << map_flag_ << std::endl;
        std::cout << "final lab " << final_lab_ << std::endl;
        std::cout << "mode " << mode_ << std::endl;
        

        // sub & pub
        sub_hesai_ = nh_.subscribe(hesai_topic_, 1, &tayoPerception::callbackHesai, this);  //64
        sub_livox_ = nh_.subscribe(sub_lidar_topic_, 1, &tayoPerception::callbackSublidar, this);  //40
        sub_status_ = nh_.subscribe(sub_status_topic_, 1, &tayoPerception::callbackSubstatus, this);

        pub_object_pcl_ = nh_.advertise<sensor_msgs::PointCloud2>(hesai_object_pcl_topic_, 1);
        pub_mapfilter_pcl_ = nh_.advertise<sensor_msgs::PointCloud2>(hesai_globalfilter_pcl_topic_, 1);
        pub_object_pt_ = nh_.advertise<sensor_msgs::PointCloud2>(hesai_object_pt_topic_, 1);
        pub_polygon_ = nh_.advertise<visualization_msgs::MarkerArray>(hesai_object_polygon_topic_, 1);
        pub_object_info_ = nh_.advertise<obstacle_msgs::obstacleArray>(hesai_object_info_topic_, 1);
        id_pub = nh_.advertise<visualization_msgs::MarkerArray>("/tayo/perception/object_id", 1);
        pub_predict = nh_.advertise<sensor_msgs::PointCloud2>("/tayo/perception/object_pred", 1);
        is_lidar_ = false;
        is_tf_ = false;
    }

    tayoPerception::~tayoPerception() {}

    void tayoPerception::initSensorInfo()
    {
        nh_.param<std::string>("perception_node/perception_param/mode", mode_, "");
        // load map data
        nh_.param<std::string>("perception_node/perception_param/map_info/image_map", map_path_, "");
        nh_.param<std::string>("perception_node/perception_param/map_info/stop_1_map", stop_1_map_path_, "");
        nh_.param<std::string>("perception_node/perception_param/map_info/stop_4_map", stop_4_map_path_, "");
        nh_.param<std::string>("perception_node/perception_param/map_info/map", map_, "");
        nh_.param<double>("perception_node/perception_param/map_info/init_pose_x", init_pose_.x, 0.0);
        nh_.param<double>("perception_node/perception_param/map_info/init_pose_y", init_pose_.y, 0.0);
        nh_.param<int>("perception_node/perception_param/map_info/map_flag", map_flag_, 2);
        nh_.param<int>("perception_node/perception_param/map_info/final_lab", final_lab_, 0);

        // load sensor topic
        nh_.param<std::string>("perception_node/sensor_info/sub_lidar1_topic", hesai_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/sub_lidar2_topic", sub_lidar_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/sub_status_topic_", sub_status_topic_, "");

        // publish topic
        nh_.param<std::string>("perception_node/sensor_info/pub_hesai_object_pcl_topic", hesai_object_pcl_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/pub_hesai_globalfilter_pcl_topic", hesai_globalfilter_pcl_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/pub_hesai_object_pt_topic", hesai_object_pt_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/pub_hesai_object_polygon_topic", hesai_object_polygon_topic_, "");
        nh_.param<std::string>("perception_node/sensor_info/pub_hesai_object_info_topic", hesai_object_info_topic_, "");

        nh_.param<std::string>("perception_node/sensor_info/map_frame_id", map_frame_id_, "");
        nh_.param<std::string>("perception_node/sensor_info/lidar_frame_id", lidar_frame_id_, "");
        nh_.param<std::string>("perception_node/sensor_info/sub_lidar_frame_id", sub_lidar_frame_id_, "");

        nh_.param<double>("perception_node/sensor_info/lidar_pose_x", sub_lidar_pose_x_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_pose_y", sub_lidar_pose_y_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_pose_z", sub_lidar_pose_z_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_ori_x", sub_lidar_ori_x_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_ori_y", sub_lidar_ori_y_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_ori_z", sub_lidar_ori_z_, 0.0);
        nh_.param<double>("perception_node/sensor_info/lidar_ori_w", sub_lidar_ori_w_, 0.0);
    }

    void tayoPerception::initPerceptionParam()
    {
        // load roi filter param
        nh_.param<double>("perception_node/perception_param/roi_filter/max_x", max_distance_.x, 200);
        nh_.param<double>("perception_node/perception_param/roi_filter/max_y", max_distance_.y, 200);
        nh_.param<double>("perception_node/perception_param/roi_filter/max_z", max_distance_.z, 0.5);

        nh_.param<double>("perception_node/perception_param/roi_filter/min_x", min_distance_.x, -200);
        nh_.param<double>("perception_node/perception_param/roi_filter/min_y", min_distance_.y, -200);
        nh_.param<double>("perception_node/perception_param/roi_filter/min_z", min_distance_.z, -5.0);

        // load map filter param
        nh_.param<double>("perception_node/perception_param/map_filter/min_thresh", min_thresh_, 0.0);
        nh_.param<double>("perception_node/perception_param/map_filter/min_long_thresh", min_long_thresh_, 0.8);
        nh_.param<double>("perception_node/perception_param/map_filter/max_thresh", max_thresh_, 2.0);
        nh_.param<double>("perception_node/perception_param/map_filter/min_dist", min_dist_, 80);
        nh_.param<double>("perception_node/perception_param/map_filter/max_dist", max_dist_, 200);

        // load ground remover param
        nh_.param<int>("perception_node/perception_param/ground_remover/grid_dim", grid_dim_, 1000);
        nh_.param<double>("perception_node/perception_param/ground_remover/per_cell", per_cell_, 0.2);
        nh_.param<double>("perception_node/perception_param/ground_remover/height_threshold", height_threshold_, 0.1);
        nh_.param<double>("perception_node/perception_param/ground_remover/close_pt_min", close_pt_min_, -1.7);
        nh_.param<double>("perception_node/perception_param/ground_remover/close_pt_max", close_pt_max_, 0.3);

        nh_.param<int>("perception_node/perception_param/noise_remover/remove_size", remove_size_, 5);

        // load euclidean cluster param
        nh_.param<double>("perception_node/perception_param/euclidean_cluster/range_cluster", range_cluster_, 0.5);
        nh_.param<int>("perception_node/perception_param/euclidean_cluster/min_cluster_size", min_cluster_size_, 10);
        nh_.param<int>("perception_node/perception_param/euclidean_cluster/max_cluster_size", max_cluster_size_, 1000);
    }

    void tayoPerception::callbackSubstatus(const decision_msgs::decisionData::ConstPtr &msg){

        status_ = *msg;
        rank_ = status_.rank;
        cur_lab_ = status_.lab;

        // std::cout << "(rank, lab) : " << rank_ <<" " << cur_lab_ << std::endl;

    }
 
    void tayoPerception::callbackHesai(const sensor_msgs::PointCloud2Ptr msg)
    {
        is_lidar_ = true;
        lidar_time_ = msg->header.stamp;
        main_lidar_raw_.clear();
        m_speed_result.markers.clear();
        // hesai raw data : hesai_pcl
        pcl::PointCloud<pcl::PointXYZI> hesai_pcl;
        pcl::fromROSMsg(*msg, hesai_pcl);

        // local to global transform
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer_.lookupTransform(map_frame_id_, lidar_frame_id_, ros::Time(0));
            is_tf_ = true;
        }
        catch (tf2::TransformException &ex)
        {
            fprintf(stderr, "%s\n", ex.what());
        }

        // global to local transform
        geometry_msgs::TransformStamped inv_transformStamped;
        try
        {
            inv_transformStamped = tf_buffer_.lookupTransform(lidar_frame_id_, map_frame_id_, ros::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            fprintf(stderr, "%s\n", ex.what());
        }

        if(is_tf_){
            ros::Time current_time = ros::Time::now();
            dt = current_time.toSec() - prev_time.toSec();
            time = current_time - prev_time;
            prev_time = current_time;
            run(transformStamped,
                inv_transformStamped,
                pub_mapfilter_pcl_,
                pub_object_pcl_,
                pub_object_pt_,
                lidar_frame_id_,
                hesai_pcl,
                dt);
            main_lidar_raw_=hesai_pcl;
           
        }
        else{
            ROS_ERROR("Fail to sub tf data");
        }
    }

    void tayoPerception::callbackSublidar(const sensor_msgs::PointCloud2Ptr msg)
    {
        if(lidar_time_.toSec() > msg->header.stamp.toSec() || !is_lidar_)
            return;

        // sub lidar cali data : sub_lidar_pcl
        pcl::PointCloud<pcl::PointXYZI> sub_lidar_pcl;
        pcl::fromROSMsg(*msg, sub_lidar_pcl);

        Eigen::Vector3f mat_pos(sub_lidar_pose_x_, sub_lidar_pose_y_, sub_lidar_pose_z_);
        Eigen::Quaternionf mat_rot(sub_lidar_ori_w_, sub_lidar_ori_x_, sub_lidar_ori_y_, sub_lidar_ori_z_);

        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3, 3>(0, 0) = mat_rot.toRotationMatrix();
        transform.block<3, 1>(0, 3) = mat_pos;

        pcl::transformPointCloud(sub_lidar_pcl, sub_lidar_pcl, transform);

        roiVelodyne(sub_lidar_pcl, sub_lidar_pcl);

        sub_lidar_pcl += main_lidar_raw_;
        geometry_msgs::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer_.lookupTransform(map_frame_id_, sub_lidar_frame_id_, ros::Time(0));
            is_tf_ = true;
        }
        catch (tf2::TransformException &ex)
        {
            fprintf(stderr, "%s\n", ex.what());
        }

        geometry_msgs::TransformStamped inv_transformStamped;
        try
        {
            inv_transformStamped = tf_buffer_.lookupTransform(sub_lidar_frame_id_, map_frame_id_, ros::Time(0));
        }
        catch (tf2::TransformException &ex)
        {
            fprintf(stderr, "%s\n", ex.what());
        }
        // sub lidar calibration check
        pubObject(pub_sublidar_pcl_, sub_lidar_pcl, sub_lidar_frame_id_);

        if(is_tf_){
            run(transformStamped,
                inv_transformStamped,
                pub_mapfilter_pcl_,
                pub_object_pcl_,
                pub_object_pt_,
                lidar_frame_id_,
                sub_lidar_pcl,
                dt
                );
        }
        else{
            ROS_ERROR("Fail to sub tf data");
        }
    }

    void tayoPerception::run(geometry_msgs::TransformStamped transformStamped,
                             geometry_msgs::TransformStamped inv_transformStamped,
                             ros::Publisher global_filtered_pub,
                             ros::Publisher ground_filtered_pub,
                             ros::Publisher object_pt_pub,
                             std::string frame_id,
                             pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                             double dt)
    {
        
        if (!cloud_in.empty())
        {
            
            clock_t start, end;

            pcl::PointCloud<pcl::PointXYZI> cloud_under, cloud_over;    // roi data
            pcl::PointCloud<pcl::PointXYZI> global_under_pcl, global_over_pcl; // global data

            pcl::PointCloud<pcl::PointXYZI> map_pcl;
            pcl::PointCloud<pcl::PointXYZI> map_under_pcl, map_over_pcl;    // map filter data
            pcl::PointCloud<pcl::PointXYZI> map_under_local_pcl, map_over_local_pcl;    // map filter global data

            pcl::PointCloud<pcl::PointXYZI> ground_filtered_pcl;    // ground remover data
            pcl::PointCloud<pcl::PointXYZI> object_grid_pcl;

            pcl::PointCloud<pcl::PointXYZI> object_pcl; // object pcl
            pcl::PointCloud<pcl::PointXYZI> object_pcl_filtered; // object pcl
            pcl::PointCloud<pcl::PointXYZI> object_pt_pcl;

            // std::vector<std::vector<cv::Point2f>> object_convex;
            std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_pc;

            geometry_msgs::Pose car_pose;
            car_pose.position.x = transformStamped.transform.translation.x;
            car_pose.position.y = transformStamped.transform.translation.y;
            car_pose.position.z = transformStamped.transform.translation.z;
            car_pose.orientation.x = transformStamped.transform.rotation.x;
            car_pose.orientation.y = transformStamped.transform.rotation.y;
            car_pose.orientation.z = transformStamped.transform.rotation.z;
            car_pose.orientation.w = transformStamped.transform.rotation.w;
            // std::cout << car_pose << std::endl;

            start = clock();
            roiFilter(cloud_in, cloud_under, cloud_over);
            end = clock();
            //ROS_INFO("Roi Object pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);
            //std::cout <<"roi size (under,upper,total) : " << cloud_under.size() << " " << cloud_over.size() << " " << cloud_in.size() << std::endl;

            if(map_flag_ == 1 || map_flag_ == 2){
                // local -> global Ï¢åÌëúÍ≥? Î≥??ôò
                start = clock();
                pcl_ros::transformPointCloud(cloud_under, global_under_pcl, transformStamped.transform);
                pcl_ros::transformPointCloud(cloud_over, global_over_pcl, transformStamped.transform);
                end = clock();
               // ROS_INFO("Transform global Object pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);

                // ?èÑÎ°? ?úÑ?ùò Í∞ùÏ≤¥ point Ï∂îÏ∂ú : map_filtered_pcl
                start = clock();
                mapFilter(global_under_pcl, map_under_pcl, map_pcl, car_pose, false);
                if(map_flag_ == 1){
                    mapFilter(global_over_pcl, map_over_pcl, map_pcl,car_pose, false);
                }
                else{
                    mapFilter(global_over_pcl, map_over_pcl, map_pcl,car_pose, true);
                }
                end = clock();
               // ROS_INFO("Filter map pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);

                // global -> local Ï¢åÌëúÍ≥? Î≥??ôò
                start = clock();
                pcl_ros::transformPointCloud(map_pcl, map_pcl,inv_transformStamped.transform);
                pcl_ros::transformPointCloud(map_under_pcl, map_under_local_pcl,inv_transformStamped.transform);
                pcl_ros::transformPointCloud(map_over_pcl, map_over_local_pcl,inv_transformStamped.transform);
                end = clock();
               // ROS_INFO("Transform local Object pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);
                //std::cout <<"map filter size (under,upper,total) : " << map_under_pcl.size() << " " << map_over_pcl.size() << " " << map_pcl.size() << std::endl;
            }
            
            // ?èÑÎ°? ?ç∞?ù¥?Ñ∞?óê?Ñú Ïß?Î©? ?†úÍ±? ?àò?ñâ
            if(map_flag_==0){
                start = clock();
                groundRemover(cloud_under,
                            ground_filtered_pcl,
                            object_grid_pcl);
                end = clock();
               // ROS_INFO("Remove ground pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);

                // object_pcl += ground_filtered_pcl;
                object_pcl += object_grid_pcl;
               // std::cout << "object pcl size " << object_pcl.size() << std::endl;
            }
            else{
                start = clock();
                groundRemover(map_under_local_pcl,
                            ground_filtered_pcl,
                            object_grid_pcl);
                end = clock();
               // ROS_INFO("Remove ground pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);

                if(map_flag_==2){
                    object_pcl += map_over_local_pcl;
                    // object_pcl += ground_filtered_pcl;
                    object_pcl += object_grid_pcl;
                   // std::cout << "object pcl size " << object_pcl.size() << std::endl;
                }
                else{
                    object_pcl += ground_filtered_pcl;
                    // object_pcl += object_grid_pcl;
                   // std::cout << "object pcl size " << object_pcl.size() << std::endl;
                }
            }

            start = clock();
            noiseRemover(object_pcl, object_pcl_filtered);
            end = clock(); 
            // ROS_INFO("Remove noise pcl : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);
            // std::cout << "remove object pcl size " << object_pcl_filtered.size() << std::endl;

            // ?ç∞?ù¥?Ñ∞ Íµ∞Ïßë?ôî ?õÑ, ÏµúÏÜå Í±∞Î¶¨ ?è¨?ù∏?ä∏ Ï∂îÏ∂ú
            start = clock();
            extractMinPt(object_pcl_filtered,
                         object_pt_pcl,
                         cluster_pc);
            end = clock();
            // ROS_INFO("Object pt : %f", (double)(end - start) / CLOCKS_PER_SEC * 1000);
            // std::cout << "size " << object_pt_pcl.size() << std::endl;

            // // ?ç∞?ù¥?Ñ∞ publish
            // pubObject(global_filtered_pub, map_under_pcl, map_frame_id_); // global
            pubObject(global_filtered_pub, map_pcl, frame_id); //local
            pubObject(ground_filtered_pub, object_pcl_filtered, frame_id);
            pubObject(object_pt_pub, object_pt_pcl, frame_id); // clustering object point cloud

            publish_polygon(cluster_pc);

            pubObjectInfo(pub_object_info_, object_pt_pcl, cluster_pc, frame_id);


            // add tracking
            track.TrackManagement(object_pt_pcl, time);
            makeID();
            id_pub.publish(m_speed_result);
             
            
        }
    }
    void tayoPerception::makeID()
    {
        //std::cout<<"makeID = " <<track.m_track.size()<<std::endl;
        pcl::PointCloud<pcl::PointXYZI> cloud;
        for(int i = 0; i < track.m_track.size(); i++)
        {
            
            if(track.m_track.at(i).life_time >= MAX_EXTRACT)
            {
                pcl::PointXYZI pt;
                pt.x = track.m_track.at(i).predict_candi.x;
                pt.y = track.m_track.at(i).predict_candi.y;
                std::string ID;
                ID = std::to_string(track.m_track[i].ID);
                ID += "\n";
                ID += "local x: ";
                ID += std::to_string(track.m_track.at(i).predict_candi.x);
                ID.erase(ID.size()-3, ID.size());
                ID += "\n";
                ID += "local y: ";
                ID += std::to_string(track.m_track.at(i).predict_candi.y);
                ID.erase(ID.size()-3, ID.size());

                visualization_msgs::Marker marker;
                marker.header.frame_id = "pos";
                marker.header.stamp = ros::Time::now();
                marker.ns = "ID";
                marker.id = i;
                marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                marker.action = visualization_msgs::Marker::ADD;

                marker.pose.position.x = track.m_track.at(i).predict_candi.x;
                marker.pose.position.y = track.m_track.at(i).predict_candi.y+1.0;
                marker.pose.position.z = 1.0;
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = 0.0;
                marker.pose.orientation.w = 1.0;

                marker.text = ID;

                marker.scale.x = 5.0;
                marker.scale.y = 5.0;
                marker.scale.z = 5.0;

                marker.color.r = 1.0f;
                marker.color.g = 1.0f;
                marker.color.b = 1.0f;
                marker.color.a = 1.0;

                marker.lifetime = ros::Duration(0.1);
                m_speed_result.markers.push_back(marker);
                cloud.push_back(pt);
            }
        }
        std::cout<<"-----------------------------"<<std::endl;
        sensor_msgs::PointCloud2 cloud_pred;
        pcl::toROSMsg(cloud, cloud_pred);
        cloud_pred.header.frame_id = "pos";
        pub_predict.publish(cloud_pred);
    }

    void tayoPerception::roiFilter(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                   pcl::PointCloud<pcl::PointXYZI> &cloud_under,
                                   pcl::PointCloud<pcl::PointXYZI> &cloud_over)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_z(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);

        *cloud = cloud_in;

        if (cloud->size() > 0)
        {
            // Create the filtering object

            //z passthrough
            pcl::PassThrough<pcl::PointXYZI> pass;
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(min_distance_.z, max_distance_.z);
            pass.setFilterLimitsNegative(false);
            pass.filter(*cloud_z);

            // 60m under
            pass.setInputCloud(cloud_z);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(min_distance_.x, max_distance_.x);
            pass.setFilterLimitsNegative(false);
            pass.filter(*cloud_filtered);

            pass.setInputCloud(cloud_filtered);
            pass.setFilterFieldName("y");
            pass.setFilterLimits(min_distance_.y, max_distance_.y);
            pass.setFilterLimitsNegative(false);
            pass.filter(*cloud_filtered);

            cloud_under = *cloud_filtered;

            //60m over
            pcl::PointCloud<pcl::PointXYZI> cloud_over_x, cloud_over_y;
            pass.setInputCloud(cloud_z);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(min_distance_.x, max_distance_.x);
            pass.setFilterLimitsNegative(true);
            pass.filter(*cloud_filtered);

            cloud_over_x = *cloud_filtered;

            pass.setInputCloud(cloud_z);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(min_distance_.x, max_distance_.x);
            pass.setFilterLimitsNegative(false);
            pass.filter(*cloud_filtered);

            pass.setInputCloud(cloud_filtered);
            pass.setFilterFieldName("y");
            pass.setFilterLimits(min_distance_.y, max_distance_.y);
            pass.setFilterLimitsNegative(true);
            pass.filter(*cloud_filtered);

            cloud_over_y = *cloud_filtered;

            cloud_over = cloud_over_x;
            cloud_over += cloud_over_y;
            
        }
    }

    void tayoPerception::roiVelodyne(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                     pcl::PointCloud<pcl::PointXYZI> &cloud_out)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);

        *cloud = cloud_in;

        if (cloud->size() > 0)
        {
            // Create the filtering object
            pcl::PassThrough<pcl::PointXYZI> pass;

            pass.setInputCloud(cloud);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(-200, 0);
            pass.setFilterLimitsNegative(false);
            pass.filter(*cloud_filtered);

            cloud_out = *cloud_filtered;

        }
    }

    cv::Mat tayoPerception::selectMap(){

        cv::Mat map;
        if(cur_lab_ == final_lab_){
            // std::cout << "final labselect" << std::endl;
            if(rank_ <= 3){
                map = stop_1_map_;
                // std::cout <<"select 1-3 stop image"<< std::endl;
            }
            else{
                map = stop_4_map_;
                // std::cout <<"select 4-6 stop image"<< std::endl;
            }
        }
        else{
            map = map_info_;
        }

        // map = stop_4_map_;
        // map = stop_1_map_;
        // map = map_info_;
        // std::cout << "(rank, lab) : " << rank_ <<" " << cur_lab_ << std::endl;

        return map;

    }

    void tayoPerception::mapFilter(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                    pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                                    pcl::PointCloud<pcl::PointXYZI> &cloud_map,
                                    geometry_msgs::Pose pose,
                                    bool is_over)
    {
        cv::Mat img = selectMap();
        // cv::Mat img = map.clone();
        int img_height = img.rows;
        int img_width = img.cols;

        for (int i = 0; i < cloud_in.size(); i++)
        {
            float diff_x = std::fabs(pose.position.x - cloud_in[i].x);
            float diff_y = std::fabs(pose.position.y - cloud_in[i].y);
            float dist = std::sqrt(std::pow(diff_x, 2) + std::pow(diff_y, 2));

            if(dist<1.3){

            }
            else{
                // ÏΩîÏä§Î≥? ?úÑÏπ? Ï¥àÍ∏∞Í∞? ?Ñ§?†ï Î∞? ?ù¥ÎØ∏Ï?? Ï¢åÌëúÍ≥? Î≥??ôò
                geometry_msgs::Point pt;

                // speedway, kiapi
                if(map_=="speedway" || map_=="kiapi"){
                    pt.x = cloud_in[i].x + init_pose_.x;
                    pt.y = cloud_in[i].y + init_pose_.y;
                    pt.z = cloud_in[i].z;

                    pt.x = pt.x;
                    pt.y = img_height - pt.y;
                }
                // ctrack
                else{
                    pt.y = cloud_in[i].x + init_pose_.x;
                    pt.x = cloud_in[i].y + init_pose_.y;
                    pt.z = cloud_in[i].z;

                    pt.y = pt.y;
                    pt.x = img_width - pt.x;
                }
                
                double z_margin;
                double dist = std::sqrt(std::pow(diff_x,2)+std::pow(diff_y, 2));
                double min_dist, max_dist;
                min_dist = min_dist_;
                max_dist = max_dist_;

                double x = (dist - min_dist)/(max_dist - min_dist);
                 if(dist>=min_dist){
                    double a= (min_long_thresh_-min_thresh_)/(std::exp(1)-1);
                    double b= min_thresh_- a;
                    z_margin = a*std::exp(x)+b;

                    z_margin = z_margin - min_thresh_;
                }
                else{
                    z_margin = 0;
                }

                if (pt.y < img_height && pt.y >= 0 && pt.x < img_width && pt.x >= 0)
                {
                    // ?èÑÎ°? ?†ïÎ≥? Ï∂îÏ∂ú : pixel
                    cv::Vec3b &pixel = img.at<cv::Vec3b>(pt.y, pt.x);

                    double pt_z;
                    double pixel_b = pixel[0];
                    double pixel_g = pixel[1];
                    double pixel_r = pixel[2];
                    if(pixel_b>=100){
                        pixel_b -= 100;
                        pt_z = pixel_b + pixel_g*0.01 + pixel_r*0.0001;
                        pt_z = -pt_z;           
                    }
                    else{
                        pt_z = pixel_b + pixel_g*0.01 + pixel_r*0.0001;
                    }

                    // road boundary filtering : Ïß?Î©? ?Ç¥Î∂??ùò pointÎß? ?Ç®Íπ?
                    if (pt_z != 0)
                    {   
                        // 60m ?ù¥?ÉÅ : z filter
                        if(is_over){
                            double z_max = pt_z + max_thresh_;
                            double z_min = pt_z + min_thresh_+z_margin;

                            if (pt.z > z_min && pt.z < z_max)
                            {
                                 cloud_out.push_back(cloud_in[i]);
                            }
                            cloud_in[i].z = pt_z+min_thresh_+z_margin;
                        }
                        // 60m ?ù¥?ïò : ground filter
                        else{
                            cloud_out.push_back(cloud_in[i]);
                        }

                        // Ïß?Î©? ?úÑ?ùò dataÎß? Ï∂îÏ∂ú
                        cloud_map.push_back(cloud_in[i]);
                        // cv::circle(img, cv::Point(pt.x, pt.y), 2, cv::Scalar(0, 0, 255), 1, -1);
                    }
                    else{
                        // cv::circle(img, cv::Point(pt.x, pt.y), 2, cv::Scalar(255, 0, 0), 1, -1);
                    }
                }
            }
        }
        // cv::imshow("map", img);
        // cv::waitKey(1);
    }

    void tayoPerception::noiseRemover(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                      pcl::PointCloud<pcl::PointXYZI> &cloud_out)
    {
        double resolution = 2.;

        std::vector<bool> leave_pt(cloud_in.size(), true);
        cv::Mat lidar_img(cv::Size(400*resolution, 400*resolution), CV_8UC1, cv::Scalar(0));

        for(int i=0; i<cloud_in.size(); i++){
            pcl::PointXYZI pt = cloud_in[i];
            pt.x = pt.x*resolution;
            pt.y = pt.y*resolution;

            cv::Point2f lidar_pt(int(400.*resolution-(pt.x+200.*resolution)), int(400.*resolution-(pt.y+200.*resolution)));
            lidar_img.at<uchar>(lidar_pt.x, lidar_pt.y) = 255;
        }

        // cv::Mat label_img;
        // cv::cvtColor(lidar_img, label_img, cv::COLOR_GRAY2BGR);

        cv::Mat img_labels, stats, centroids;
        int numOfLables = connectedComponentsWithStats(lidar_img, img_labels, stats, centroids, 8, CV_32S); //labeling

        int num = 1;
        std::vector<cv::Point2f> labels;
        for (int j = 1; j < numOfLables; j++) {
            int area = stats.at<int>(j, cv::CC_STAT_AREA);
            int left = stats.at<int>(j, cv::CC_STAT_LEFT);
            int top = stats.at<int>(j, cv::CC_STAT_TOP);
            int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);

            if (area <= remove_size_) { //?ùºÎ≤®ÎßÅ Î©¥Ï†Å ?ôï?ù∏
                // cv::rectangle(label_img, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 0, 255), 1);
                // cv::putText(label_img, std::to_string(area), cv::Point(left + 20, top + 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);

                for(int x=left-1; x<left+width+2; x++){
                    for(int y=top-1; y<top+height+2; y++){
                        cv::Point2f pt;
                        pt.x = (200.*resolution - y)/resolution;
                        pt.y = (200.*resolution - x)/resolution;

                        for(int i=0; i<cloud_in.size(); i++){
                            double dist = std::sqrt(std::pow(pt.x-cloud_in[i].x,2)+std::pow(pt.y-cloud_in[i].y,2));
                            if(dist<=0.5){
                                leave_pt[i]=false;
                            }
                        }
                    }
                }

                num++;
            }
        }

        // cv::imshow("lable", label_img);
        // cv::waitKey(1);

        // rain
        for(int i=0; i<cloud_in.size(); i++){
            if(leave_pt[i] == true){
                // if(cloud_in[i].x<0 && cloud_in[i].x>-15. && std::abs(cloud_in[i].y)<=2.0){
                // }
                // else{
                    cloud_out.push_back(cloud_in[i]);
                // }
            }
        }
    }

    void tayoPerception::groundRemover(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                       pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                                       pcl::PointCloud<pcl::PointXYZI> &cloud_grid_out)
    {
        // set the exact point cloud size -- the vectors should already have enough
        // space
        int num_point = cloud_in.size();
        // Grid Î≥??ôò ?ùº?ù¥?ã§ point ÏµúÏÜå/ÏµúÎ?? zÍ∞? ÎπÑÍµêÎ•? ?úÑ?ïú Î≥??àò ?Éù?Ñ±

        float min[grid_dim_][grid_dim_];
        float max[grid_dim_][grid_dim_];
        bool init[grid_dim_][grid_dim_];
        std::vector<std::vector<std::vector<int>>> indices;
        indices.resize(grid_dim_);

        for (int i = 0; i < grid_dim_; i++)
        {
            memset(min[i], 0.0f, grid_dim_);
            memset(max[i], 0.0f, grid_dim_);
            memset(init[i], false, grid_dim_);
            indices[i].resize(grid_dim_);
        }

        per_cell_ = max_distance_.x*2/grid_dim_;
        // ?Üí?ù¥Í∏∞Î∞ò Grid Map ?Éù?Ñ±
        for (unsigned i = 0; i < num_point; ++i)
        {
            int x = ((grid_dim_ / 2) + cloud_in[i].x / per_cell_);
            int y = ((grid_dim_ / 2) + cloud_in[i].y / per_cell_);

            if (x >= 0 && x < grid_dim_ && y >= 0 && y < grid_dim_)
            {
                if (!init[x][y])
                {
                    // Ï¥àÍ∏∞Í∞? ?Éù?Ñ±
                    min[x][y] = cloud_in[i].z;
                    max[x][y] = cloud_in[i].z;

                    init[x][y] = true;
                    indices[x][y].push_back(i);
                }
                else
                {
                    // Grid?óê ?è¨?ï®?êú zÍ∞? min, max ÎπÑÍµê
                    min[x][y] = std::min(min[x][y], cloud_in[i].z);
                    max[x][y] = std::max(max[x][y], cloud_in[i].z);
                    indices[x][y].push_back(i);
                }
            }
        }
        // Grid Cell?ùò [ÏµúÎ?? ?Üí?ù¥ - ÏµúÏÜå ?Üí?ù¥] Ï∞®Î?? ?Üµ?ï¥ Ïß?Î©¥ÏùÑ ?†ú?ô∏?ïú Í∞ùÏ≤¥ ?õÑÎ≥?
        // ?òÅ?ó≠ Ï∂îÏ∂ú
        for (int x = 0; x < grid_dim_; x++)
        {
            for (int y = 0; y < grid_dim_; y++)
            {
                // height_diff_threshold_ : ÏµúÎ?? ?Üí?ù¥-ÏµúÏÜå ?Üí?ù¥ ?ûÑÍ≥ÑÍ∞í
                if (fabs(max[x][y] - min[x][y]) > height_threshold_ 
                && init[x][y] == true)
                {
                    double mid_z = (max[x][y]+min[x][y])/2.;
                    if(mid_z >= -1.7){
                        for (int k = 0; k < indices[x][y].size(); k++)
                        {
                            if(mid_z <= cloud_in[indices[x][y][k]].z){
                                pcl::PointXYZI pt = cloud_in[indices[x][y][k]];
                                // pt.z = 0.0;

                                cloud_out.push_back(pt);
                            }
                        }

                        pcl::PointXYZI pt = cloud_in[indices[x][y][0]];
                        // pt.z = 0.0;
                        cloud_grid_out.push_back(pt);
                    }
                }
                else if(min[x][y]>close_pt_min_ && max[x][y]<close_pt_max_
                        && init[x][y] == true)
                {
                    if(std::abs(cloud_in[indices[x][y][0]].x) <= 5. && std::abs(cloud_in[indices[x][y][0]].y) <= 5.){
                        double mid_z = (max[x][y]+min[x][y])/2.;
                        for (int k = 0; k < indices[x][y].size(); k++)
                        {
                            if(mid_z <= cloud_in[indices[x][y][k]].z){
                                if(cloud_in[indices[x][y][k]].x<=3 && cloud_in[indices[x][y][k]].x>=-4.
                                    && std::abs(cloud_in[indices[x][y][k]].y)<=1.1){

                                }
                                else{
                                    cloud_out.push_back(cloud_in[indices[x][y][k]]);
                                }
                            }

                        }
                        pcl::PointXYZI pt = cloud_in[indices[x][y][0]];
                        if(pt.x <= 3 && pt.x >= -4. && std::abs(pt.y) <= 1.1){
                        }
                        else{
                            // pt.z = 0.0;
                            cloud_grid_out.push_back(pt);
                        }
                    }
                }

            }
        }
    }

    void tayoPerception::matchCluster(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                      pcl::PointCloud<pcl::PointXYZI> &cloud_out)
    {

        int debug=0;

        for(int i=0; i<cloud_in.size(); i++){
            double min_dist = 10000;
            int min_index = -1;

            for(int j=0; j<cloud_in.size(); j++){
                if(i!=j){
                    double x_diff = std::abs(cloud_in[i].x-cloud_in[j].x);
                    double y_diff = std::abs(cloud_in[i].y-cloud_in[j].y);
                    double dist = std::sqrt(std::pow(x_diff,2)+std::pow(y_diff,2));

                    // std::cout << i << "  " << j <<" " << dist << std::endl;
                    if(dist < min_dist){
                        min_dist = dist;
                        min_index=j;
                    }
                }
            }

            // std::cout << min_dist << std::endl;
            if(min_dist <= 3.0){
                double cur_dist = std::sqrt(std::pow(cloud_in[i].x,2)+std::pow(cloud_in[i].y,2));
                double match_dist = std::sqrt(std::pow(cloud_in[min_index].x,2)+std::pow(cloud_in[i].y,2));

                // std::cout << cur_dist  << " " << match_dist <<std::endl;
                if(cur_dist < match_dist){
                    cloud_out.push_back(cloud_in[i]);
                    // std::cout << cloud_in[i] << std::endl;
                }
                // else{
                //     cloud_out.push_back(cloud_in[min_index]);
                //     std::cout << cloud_in[min_index] << std::endl;
                // }
            }
            else{
                cloud_out.push_back(cloud_in[i]);
                // std::cout << cloud_in[i] << std::endl;
            }
        }
        // if(cloud_in.size()>=6){
        //     std::cout <<"debug" <<std::endl;
        //     debug++;
        // }

    }

    void tayoPerception::extractMinPt(pcl::PointCloud<pcl::PointXYZI> &cloud_in,
                                      pcl::PointCloud<pcl::PointXYZI> &cloud_out,
                                      std::vector<pcl::PointCloud<pcl::PointXYZI>> &cloud_out_pcl)
    {
        
        pcl::PointCloud<pcl::PointXYZI> cluster_pt;
        std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_pc;
        
        // Data Clustering
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(cloud_in.makeShared());

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        // ?†êÍ≥? ?†ê?Ç¨?ù¥?ùò Í±∞Î¶¨
        ec.setClusterTolerance(range_cluster_);
        // Íµ∞Ïßë?ôî ÏµúÏÜå Point Í∞??àò
        ec.setMinClusterSize(min_cluster_size_);
        // Íµ∞Ïßë?ôî ÏµúÎ?? Point Í∞??àò
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_in.makeShared());
        ec.extract(cluster_indices);

        int num = 0;
        // Íµ∞Ïßë?ôî ?êú ?è¨?ù∏?ä∏ ?Å¥?ùº?ö∞?ìú Î∂ÑÎ•ò
        for (std::vector<pcl::PointIndices>::const_iterator it =
                 cluster_indices.begin();
             it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZI> cluster;
            for (std::vector<int>::const_iterator pit = it->indices.begin();
                 pit != it->indices.end(); ++pit)
            {
                pcl::PointXYZI pt = cloud_in.points[*pit];
                cluster.push_back(pt);
                
            }
            // Íµ∞Ïßë?ôî ?êú ?è¨?ù∏?ä∏ ?Å¥?ùº?ö∞?ìú?ùò Ï§ëÏ†ê?ùÑ Ï∂úÎ†• ?è¨?ù∏?ä∏Î°? ?Ñ§?†ï
            double min_dis = DBL_MAX;
            int min_idx = -1;
            double x_max = -1000;
            double x_min = 1000;
            double y_max = -1000;
            double y_min = 1000;
            double cluster_w, cluster_h;
            int cluster_cnt=0;

            pcl::PointCloud<pcl::PointXYZI> pc;
            for (int i = 0; i < cluster.size(); i++)
            {
                if(x_max < cluster[i].x ){
                    x_max = cluster[i].x;
                }
                if(x_min > cluster[i].x ){
                    x_min = cluster[i].x;
                }
                if(y_max < cluster[i].y){
                    y_max = cluster[i].y;
                }
                if(y_min > cluster[i].y){
                    y_min = cluster[i].y;
                }

                cluster_h = std::fabs(x_max - x_min);
                cluster_w = std::fabs(y_max - y_min);
                cluster_cnt+=1;

                pc.push_back(cluster[i]);

                double dis = sqrt(pow(cluster[i].x, 2) + pow(cluster[i].y, 2));
                if (dis < min_dis)
                {
                    min_dis = dis;
                    min_idx = i;
                }
            }

            cluster_pc.push_back(pc);
            cluster_pt.push_back(cluster[min_idx]);

            // Ï∞? Ï£ºÎ???óê Ï∞çÌûå clustering ?†úÍ±?
            // pcl::PointCloud<pcl::PointXYZI> result;
            // for (int i = 0; i < cloud_out.size(); i++)
            // {
            //     double dis = sqrt(pow(cloud_out[i].x, 2) + pow(cloud_out[i].y, 2));
            //     if (dis >= 2.0)
            //     {
            //         result.push_back(cloud_out[i]);
            //     }
            // }
            // cloud_out.clear();
            // cloud_out = result;
        }

        matchCluster(cluster_pt, cloud_out);
        if(cluster_pt.size() == cloud_out.size()){
            cloud_out_pcl = cluster_pc;
        }
        else{
            for(int i=0; i<cluster_pt.size(); i++){
                for(int j=0; j<cloud_out.size(); j++){
                    if(cluster_pt[i].x == cloud_out[j].x && cluster_pt[i].y == cloud_out[j].y){
                        cloud_out_pcl.push_back(cluster_pc[i]);
                    }
                }
            }
            // std::cout << "cluster filtering " << cluster_pt.size() << " => " << cloud_out.size() << std::endl;
        }

        // cloud_out=cluster_pt;
        // cloud_out_pcl = cluster_pc;
    }

    void tayoPerception::pubObject(ros::Publisher pub,
                                   pcl::PointCloud<pcl::PointXYZI> cloud_in,
                                   std::string frame_id)
    {
        sensor_msgs::PointCloud2 pcl_msg;
        pcl::toROSMsg(cloud_in, pcl_msg);
        pcl_msg.header.frame_id = frame_id;
        pub.publish(pcl_msg);
    }

    void tayoPerception::publish_polygon(std::vector<pcl::PointCloud<pcl::PointXYZI>> point_cloud){
        
        visualization_msgs::MarkerArray object_polygon;

        for(int i=0; i<point_cloud.size(); i++){

            visualization_msgs::Marker polygon;
            makePolygon(point_cloud[i], 1, polygon);
            object_polygon.markers.push_back(polygon);
        }

        pub_polygon_.publish(object_polygon);
    }


    void tayoPerception::pubObjectInfo(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZI> obj_cluster, std::vector<pcl::PointCloud<pcl::PointXYZI>> point_cloud, std::string frame_id)
    {
        obstacle_msgs::obstacleArray arr;
        for(int i = 0; i < obj_cluster.size(); i++){
            obstacle_msgs::obstacle obj;

            pcl::PointXYZI minPt = obj_cluster[i];

            visualization_msgs::Marker polygon;
            makePolygon(point_cloud[i], i, polygon);

            obj.x = minPt.x;
            obj.y = minPt.y;
            obj.z = minPt.z;

            obj.polygon = polygon;

            arr.obstacles.push_back(obj);
        }
        
        pub.publish(arr);
        // std::cout << "pub object info" << arr.obstacles.size() << std::endl;
    }

    void tayoPerception::makePolygon(pcl::PointCloud<pcl::PointXYZI> point_cloud, int id, visualization_msgs::Marker &polygon)
    {
        std::vector<cv::Point2f> arr;    
        for(int i = 0; i < point_cloud.size(); i++){
            cv::Point2f pt;
            pt.x = point_cloud[i].x;
            pt.y = point_cloud[i].y;

            arr.push_back(pt);
        }

        std::vector<cv::Point2f> hull(arr.size());
        cv::convexHull(arr, hull);

        polygon.header.stamp = ros::Time::now();
        polygon.header.frame_id = "pos";
        polygon.ns = "polygon";
        polygon.id = id;
        polygon.type = visualization_msgs::Marker::LINE_STRIP;

        int hullSize = hull.size();
        geometry_msgs::Point p[hullSize + 1];
        polygon.points.resize(hullSize + 1);

        for(int i = 0; i < hullSize; i++){
            polygon.points[i].x = hull[i].x;
            polygon.points[i].y = hull[i].y;
            polygon.points[i].z = 0.0;
        }
        polygon.points[hullSize].x = hull[0].x;
        polygon.points[hullSize].y = hull[0].y;
        polygon.points[hullSize].z = 0.0;

        polygon.scale.x = 0.1;
        polygon.color.a = 1.0;
        polygon.color.r = 1.0;
        polygon.color.g = 0.0;
        polygon.color.b = 0.0;
    }


    void tayoPerception::makeObjPt(pcl::PointCloud<pcl::PointXYZI> point_cloud, pcl::PointXYZI &minPt){
        double min_dis = DBL_MAX;
        int iter = -1;
        for(int i = 0; i < point_cloud.size(); i++){
            double dis = sqrt(pow(point_cloud[i].x, 2) + pow(point_cloud[i].y, 2));

            if(min_dis > dis){
                min_dis = dis;
                iter = i;
            }
        }

        minPt = point_cloud[iter];
    }
}