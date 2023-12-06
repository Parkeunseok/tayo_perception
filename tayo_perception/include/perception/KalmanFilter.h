#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include<iostream>
#include<math.h>
#include<cmath>
#include<stack>
#include<vector>
#include<Eigen/Dense>
#include <opencv2/opencv.hpp>
class KalmanFilter
{
public:
    KalmanFilter(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& H,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
		const Eigen::MatrixXd& P);

    //Function
    /**
	* Initialize the filter with initial states as zero.
	*/
    void Init();
    /**
	* Initialize the filter with a guess for initial states.
	*/
    void Init(const Eigen::VectorXd& x0);
	/**
	* Set Matrix
	*/
	void Set_matrix(
			const Eigen::MatrixXd& A,
			const Eigen::MatrixXd& B,
			const Eigen::MatrixXd& H,
			const Eigen::MatrixXd& Q,
			const Eigen::MatrixXd& R,
			const Eigen::MatrixXd& P);
    /**
	* Update the prediction based on control input.
	*/
    void Predict(const Eigen::VectorXd& u);
    /**
	* Update the estimated state based on measured values.
	*/
    void Update(const Eigen::VectorXd& y, const Eigen::MatrixXd& R);
    /**
	* Update the dynamics matrix.
	*/
	void Update_dynamics(const Eigen::MatrixXd A);
	/**
	* Update the output matrix.
	*/
	void Update_output(const Eigen::MatrixXd C);
    /**
    *return the current state
    */
    Eigen::VectorXd state() { return x_hat; };
	Eigen::MatrixXd P;
private:

	// Matrices for computation
	Eigen::MatrixXd A, B, H, Q, R, K, P0;

	// System dimensions
	int m, n, c;

	// Is the filter initialized?
	bool initialized = false;

	// n-size identity
	Eigen::MatrixXd I;
	
	// Estimated states
	Eigen::VectorXd x_hat;

};
#endif // KALMANFILTER_H
