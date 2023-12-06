#include "perception/tayoAssocation.h"


float euclideanDist(cv::Point2f &p, cv::Point2f &q)
{
    cv::Point2f a;
    a.x= p.x;
    a.y = p.y;

    cv::Point2f b;
    b.x = q.x;
    b.y = q.y;

    cv::Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void Track::setupKalman()
{
    A << 1, 0, 0, 0,
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 0, 1;

    B << 0, 0,
        0, 0,
        0, 0,
        0, 0;
   
    H << 1, 0, 0, 0,
        0, 0, 1, 0;

    Q << 0.1, 0, 0 ,0, 
        0, 0.1, 0, 0,
        0, 0, 0.1, 0,
        0, 0, 0, 0.1;

    R << 0.01, 0,
        0, 0.01;

    P << 0.05, 0, 0, 0,
        0, 0.05, 0, 0,
        0, 0, 0.05, 0,
        0, 0, 0, 0.05;

    std::cout<< "Kalman algorithm"<<std::endl;
}


Track::Track() : kf(A, B, H, Q, R, P)
{
    setupKalman();
    kf.Set_matrix(A,B,H,Q,R,P);
    kf.Init();
    life_time=-1;
}

Track::~Track()
{

}
                                                                                                                                                                                                 
void Track::unmatchedTrack()
{
    if(life_time > -1)
    {
        life_time--;
        
        Eigen::VectorXd u(c);
        u << 0, 0;
        kf.Predict(u);
        z_candi.x = kf.state()(0,0);
        z_candi.y = kf.state()(2,0);
        predict_candi.x = kf.state()(0,0);
        predict_candi.y = kf.state()(2,0);

        Eigen::VectorXd t(m);
        t << z_candi.x, z_candi.y;
       
        R << 0.01, 0,
              0, 0.01;
        kf.Update(t, R);
        fore_candi.x = kf.state()(0,0);
        fore_candi.y = kf.state()(2,0);
    }
    if(life_time == -1)
        return;
}

void Track::tracking(cv::Point2f obj, double dt)
{
    std::cout << "tracking"<< std::endl;
    prev_pos.push_back(obj);
    cur_pos = obj;
    // POSITION_X, VEL_X, POSITION_Y, VEL_Y
    A << 1, dt, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, dt,
        0, 0, 0, 1;

    kf.Update_dynamics(A);
    if (life_time == -1)
    {
        ID = TrackID++;
        life_time = 1;
        z_candi = obj;
        fore_candi = obj;
        Eigen::VectorXd X(4);
        
        X<<obj.x,
           1.0,
           obj.y,
           1.0;

        kf.Init(X);
        Eigen::VectorXd u(c);
        u << 0, 0;

        kf.Predict(u);
        predict_candi.x = kf.state()(0,0);
        predict_candi.y = kf.state()(2,0);
    }
    else
    {
        z_candi = obj;
        
        Eigen::VectorXd u(c);
        u << 0, 0;

        Eigen::VectorXd t(m);
        t << z_candi.x, z_candi.y;
 
        R << 0.01, 0,
              0, 0.01;
        kf.Update(t,R);
        kf.Predict(u);
        predict_candi.x = kf.state()(0,0);
        predict_candi.y = kf.state()(2,0);

        fore_candi.x = kf.state()(0,0);
        fore_candi.y = kf.state()(2,0);
        if (life_time < MAXLIFETIME)
        {
            life_time++;
        }
    }
}



void HungarianAlgorithm::trackCostFunction(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float dist_th)
{
    std::cout<< "Cost function #1"<<" "<<n_size<<std::endl;
    float x_max=0.0;
    for(int i=0 ; i<n_size ; i++)
    {
        for(int j=0 ; j<n_size ; j++)
        {
            if(j>=measure.size())
            {
                cost[i][j] = 255;
                init_cost[i][j] = 255;
                x_max = std::max(x_max, cost[i][j]);
            }
            else
            {
                cv::Point2f predict_candi_pt = vc_tracks[i].predict_candi;
                cv::Point2f candi_pt = measure[j];
                float dist = euclideanDist(predict_candi_pt,candi_pt);
              
                if(dist >= dist_th)
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    x_max = std::max(x_max, cost[i][j]);
                }
                else
                {
                    cost[i][j] = dist;
                    init_cost[i][j] = dist;
                    x_max = std::max(x_max, cost[i][j]);
                }
            }
        }
    }
    
    for (int i = 0; i < n_size; i++)
    {
        for (int j = 0; j < n_size; j++)
        {
            cost[i][j] = x_max - cost[i][j];
        }
    }

    for (int i = 0; i < n_size; i++)
    {
        for (int j = 0; j < n_size; j++)
        {
            std::cout << cost[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void HungarianAlgorithm::measureCostFunction(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float dist_th)
{
    std::cout<< "Cost function #2"<<" "<<n_size<< std::endl;
    float x_max=0.0;
    for(int i=0 ; i<n_size ; i++)
    {
        for(int j=0 ; j<n_size ; j++)
        {
            if(i>=vc_tracks.size())
            {
                cost[i][j] = 255;
                init_cost[i][j] = 255;
                x_max = std::max(x_max, cost[i][j]);
            }
            else
            {
                cv::Point2f predict_candi_pt = vc_tracks[i].predict_candi;
                cv::Point2f candi_pt = measure[j];
                float dist = euclideanDist(predict_candi_pt,candi_pt);
                
                if(dist >= dist_th)
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    x_max = std::max(x_max, cost[i][j]);
                }
                else
                {
                    cost[i][j] = dist;
                    init_cost[i][j] = dist;
                    x_max = std::max(x_max, cost[i][j]);
                }
            }
        }
    }
    for (int i = 0; i < n_size; i++)
    {
        for (int j = 0; j < n_size; j++)
        {
            cost[i][j] = x_max - cost[i][j];
        }
    }

    for (int i = 0; i < n_size; i++)
    {
        for (int j = 0; j < n_size; j++)
        {
            std::cout << cost[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void HungarianAlgorithm::init_labels()
{
    // step #1
    memset(label_x, 0, sizeof(label_x));      // x = measurement
    memset(label_y, 0, sizeof(label_y));      // y label은 모두 0으로 초기화.  == track

    for (int x = 0; x < n_size; x++)
        for (int y = 0; y < n_size; y++)
            label_x[x] = std::max(label_x[x], cost[x][y]);    // cost중에 가장 큰 값을 label 값으로 잡음.
}

void HungarianAlgorithm::update_labels()
{
    float delta = (float)INF;

    // slack통해서 delta값 계산함.
    for (int y = 0; y < n_size; y++)
        if (!T[y]) delta = std::min(delta, slack[y]);

    for (int x = 0; x < n_size; x++)
        if (S[x]) label_x[x] -= delta;
    for (int y = 0; y < n_size; y++) {
        if (T[y]) label_y[y] += delta;
        else slack[y] -= delta;
    }
}

void HungarianAlgorithm::add_to_tree(int x, int parent_x)
{
    S[x] = true;            // S집합에 포함.
    parent[x] = parent_x;   // augmenting 할때 필요.

    for (int y = 0; y < n_size; y++) {                                   // 새 노드를 넣었으니, slack 갱신해야함.
        if (label_x[x] + label_y[y] - cost[x][y] < slack[y]) {
            slack[y] = label_x[x] + label_y[y] - cost[x][y];
            slackx[y] = x;
        }
    }
}

void HungarianAlgorithm::augment()
{
    if (Match_num == n_size) return;
    int root;   // 시작지점.
    std::queue<int> q;

    memset(S, false, sizeof(S)); // S = my current vertex
    memset(T, false, sizeof(T)); // T = S neighbors
    memset(parent, -1, sizeof(parent)); // previous vertex

    // root를 찾음. 아직 매치안된 y값을 찾음ㅇㅇ.
    for (int x = 0; x < n_size; x++)
    {
        if (xMatch[x] == -1)
        {
            q.push(root = x);
            parent[x] = -2;
            S[x] = true;
            break;
        }
    }

    // slack 초기화.
    for (int y = 0; y < n_size; y++)
    {
        slack[y] = label_x[root] + label_y[y] - cost[root][y];  // slack  == fesible label 
        slackx[y] = root; 
    }

    int x, y;
    // augment function
    while (true)
    {
        // bfs cycle로 tree building.
        while (!q.empty())
        {
            x = q.front(); q.pop();
            for (y = 0; y < n_size; y++)
            {
                if (cost[x][y] == label_x[x] + label_y[y] && !T[y])
                {
                    if (yMatch[y] == -1) break;
                    T[y] = true;
                    q.push(yMatch[y]);
                    add_to_tree(yMatch[y], x);
                }
            }
            if (y < n_size) break;
        }
        if (y < n_size) break;

        while (!q.empty()) q.pop();

         update_labels(); // 증가경로가 없다면 label 향상ㄱ.

        // label 향상을 통해서 equality graph의 새 edge를 추가함.
        // !T[y] && slack[y]==0 인 경우에만 add 할 수 있음.
        for (y = 0; y < n_size; y++)
        {
            if (!T[y] && slack[y] == 0)
            {
                if (yMatch[y] == -1)
                {          // 증가경로 존재.
                    x = slackx[y];
                    break;
                }
                else
                {
                    T[y] = true;
                    if (!S[yMatch[y]])
                    {
                        q.push(yMatch[y]);
                        add_to_tree(yMatch[y], slackx[y]);
                    }
                }
            }
        }
        if (y < n_size) break;  // augment path found;
    }

    if (y < n_size)
    {        
        // augment path exist
        Match_num++;

        for (int cx = x, cy = y, ty; cx != -2; cx = parent[cx], cy = ty)
        {
            ty = xMatch[cx];
            yMatch[cy] = cx;
            xMatch[cx] = cy;
        }
        augment();  // 새 augment path 찾음.
    }
}

void HungarianAlgorithm::hungarian()
{
    Match_num=0;
    memset(xMatch, -1, sizeof(xMatch));
    memset(yMatch, -1, sizeof(yMatch));
    init_labels();
    augment();

    for(int i = 0 ; i < n_size; i++)
        std::cout << "Hungarian = " << xMatch[i] << " "<< yMatch[xMatch[i]] <<std::endl;
}

void HungarianAlgorithm::AssociaTion(std::vector<Track> &vc_tracks, std::vector<cv::Point2f> &measure, float DIST_TH, double dt)
{
    std::cout << "AssociaTion" << std::endl;
    std::cout << "Track" << vc_tracks.size()<< std::endl;
    std::cout << "measure" << measure.size()<< std::endl;
    if(vc_tracks.size() > measure.size())
    {
        n_size = vc_tracks.size();
        trackCostFunction(vc_tracks, measure, DIST_TH);
        hungarian();

        // add study
        for (int x = 0; x < n_size; x++)
        {
            if (init_cost[x][xMatch[x]] > DIST_TH) //  이하이면
            {
                vc_tracks[x].unmatchedTrack();  
                if (xMatch[x] /* measure index */  < measure.size()) // 둘다 유효할떄 track match measurment index
                {   
                    Track new_track;
                    new_track.tracking(measure[xMatch[x]], dt);
                    vc_tracks.push_back(new_track);
                }
            }
            else
            {
                vc_tracks[x].tracking(measure[xMatch[x]], dt);
            }
        }
    }
    else
     {
        n_size  = measure.size();
        measureCostFunction(vc_tracks, measure, DIST_TH);
        hungarian();

        for (int y = 0; y < n_size; y++)
        {
            if (init_cost[yMatch[y]][y] > DIST_TH)
            {
                if (yMatch[y] /*track index */ < vc_tracks.size()) // 둘다 유효할때 // mesurement match track index
                {
                    vc_tracks[yMatch[y]].unmatchedTrack(); 
                }
                Track new_track;
                new_track.tracking(measure[y], dt);
                vc_tracks.push_back(new_track);
            }
            else
            {
                vc_tracks[yMatch[y]].tracking(measure[y], dt);
            }
        }
     }
}

std::vector<cv::Point2f> tayoAssociation::pointcloudTovector(pcl::PointCloud<pcl::PointXYZI> pt)
{
    std::vector<cv::Point2f> cloud;
    for(int i = 0; i < pt.size(); i++)
    {
        cv::Point2f point;
        point.x = pt.at(i).x;
        point.y = pt.at(i).y;
        cloud.push_back(point);
    }
    return cloud;
}

void tayoAssociation::TrackManagement(pcl::PointCloud<pcl::PointXYZI> &measure, ros::Duration run_time)
{
    m_time = run_time.toSec();
    std::vector<cv::Point2f> m_measure = pointcloudTovector(measure);
    if(m_track.size()==0)
    {
        std::cout << "track init" << std::endl;
        for (int i = 0; i < measure.size(); i++)
        {
            Track newTrack;
            cv::Point2f pt;
            pt.x = measure.at(i).x;
            pt.y = measure.at(i).y;
            newTrack.tracking(pt, m_time);
            m_track.push_back(newTrack);
        }
    }
    else
    {
        float min_cost = 10.f;
        if (measure.size() == 1 && m_track.size() == 1)
        {
            std::cout << "track object cnt 1" << std::endl;
            cv::Point2f init_measure_pt;
            init_measure_pt.x = measure.at(0).x;
            init_measure_pt.y = measure.at(0).y;
            cv::Point2f init_predict_pt = m_track[0].predict_candi;
            float cost = euclideanDist(init_measure_pt, init_predict_pt);

            if(cost > min_cost)
            {
                Track newTrack;
                cv::Point2f track_pt;
                track_pt.x = measure.at(0).x;
                track_pt.y = measure.at(0).y;
                newTrack.tracking(track_pt, m_time);
                m_track.push_back(newTrack);
                // add study
                m_track[0].unmatchedTrack();
                if(m_track[0].life_time == -1)
                   m_track.erase(m_track.begin());
            }
            else
            {
                m_track[0].tracking(init_measure_pt, m_time);
            }
        }
        else
        {
            std::cout << "track multiple object " << std::endl;
           
            HungarianAlgorithm hungarian;
            hungarian.AssociaTion(m_track, m_measure, min_cost, m_time);

            for (int i = 0; i < m_track.size(); i++)
            {
                if (m_track[i].life_time == -1)
                {
                    m_track.erase(m_track.begin() + i);
                    i--;
                }
            }
        }
    }
}

HungarianAlgorithm::HungarianAlgorithm()
{

}

HungarianAlgorithm::~HungarianAlgorithm()
{

}