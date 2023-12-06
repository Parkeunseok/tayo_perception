#ifndef TAYOKALMANFILTER_H
#define TAYOKALMANFILTER_H
#include<iostream>
#include<math.h>
#include<cmath>
#include<stack>
#include<vector>
#include<Eigen/Dense>
#include <opencv2/opencv.hpp>

class tayoKalmanFilter
{
    Eigen::MatrixXd A, B, H, Q, R, K, P0;
    Eigen::MatrixXd P;
    int m, n, c;
    bool initialized = false;
    Eigen::MatrixXd I;
    Eigen::MatrixXd x_hat;
public:
    tayoKalmanFilter();
    tayoKalmanFilter(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& H,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
		const Eigen::MatrixXd& P);

    void Init();
    void Init(const Eigen::MatrixXd& x0);
    void Set_matrix(
			const Eigen::MatrixXd& A,
			const Eigen::MatrixXd& B,
			const Eigen::MatrixXd& H,
			const Eigen::MatrixXd& Q,
			const Eigen::MatrixXd& R,
			const Eigen::MatrixXd& P);
    void Predict(const Eigen::MatrixXd& u);
    void Update(const Eigen::MatrixXd& y, const Eigen::MatrixXd& R);
    void Update_dynamics(const Eigen::MatrixXd A);
    Eigen::MatrixXd state() { return x_hat; };
    ~tayoKalmanFilter();
};
#endif TAYOKALMANFILTER_H