#include<iostream>
#include<math.h>
#include<cmath>
#include<stack>
#include<vector>
#include<Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "perception/tayoKalmanFilter.h"
tayoKalmanFilter::tayoKalmanFilter(
    	const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& H,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
		const Eigen::MatrixXd& P)
    : A(A), B(B), H(H), Q(Q), R(R), P0(P),
		m(H.rows()), n(A.rows()), c(B.cols()), initialized(false),
		I(n, n), x_hat(n, n)
{
    I.setIdentity();
}

void tayoKalmanFilter::Init(const Eigen::MatrixXd& x0) 
{
	x_hat = x0;
	std::cout<< x_hat.rows()<<x_hat.cols()<<std::endl;
	P = P0;
	std::cout<< P.rows()<<P.cols()<<std::endl;
	initialized = true;
}

void tayoKalmanFilter::Init() 
{
	x_hat.setIdentity();	
	
	P = P0;
	
	initialized = true;
}

void tayoKalmanFilter::Set_matrix(
			const Eigen::MatrixXd& A,
			const Eigen::MatrixXd& B,
			const Eigen::MatrixXd& H,
			const Eigen::MatrixXd& Q,
			const Eigen::MatrixXd& R,
			const Eigen::MatrixXd& P)
{
	this->A = A;
	this->B = B;
	this->H = H;
	this->Q = Q;
	this->R = R;
	this->P0 = P;
}

void tayoKalmanFilter::Predict(const Eigen::MatrixXd& u) 
{
	if(!initialized) 
	{
		std::cout << "Filter is not initialized! Initializing with trivial state.";
		Init();
	}
	x_hat = A*x_hat + B*u;
	P = A*P*A.transpose() + Q;
}


void tayoKalmanFilter::Update(const Eigen::MatrixXd& y, const Eigen::MatrixXd& R) 
{
	K = P*H.transpose()*(H*P*H.transpose() + R).inverse();
	x_hat += K * (y - H*x_hat);
	std::cout << "======================"<<std::endl;
	std::cout << "x_hat: " << x_hat(0,0) << std::endl;
	std::cout << "x_hat: " << x_hat(1,1) << std::endl;
	std::cout << "x_hat: " << x_hat(2,2) << std::endl;
	std::cout << "x_hat: " << x_hat(3,3) << std::endl;
	std::cout << "======================"<<std::endl;
	P = (I - K*H)*P;	
}

void tayoKalmanFilter::Update_dynamics(const Eigen::MatrixXd A) 
{
	this->A = A;
}

tayoKalmanFilter::tayoKalmanFilter()
{
    
}

tayoKalmanFilter::~tayoKalmanFilter()
{
    
}