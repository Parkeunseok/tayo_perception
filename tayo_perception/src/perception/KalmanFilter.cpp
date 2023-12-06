#include "perception/KalmanFilter.h"

KalmanFilter::KalmanFilter(
    	const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& B,
		const Eigen::MatrixXd& H,
		const Eigen::MatrixXd& Q,
		const Eigen::MatrixXd& R,
		const Eigen::MatrixXd& P)
    : A(A), B(B), H(H), Q(Q), R(R), P0(P),
		m(H.rows()), n(A.rows()), c(B.cols()), initialized(false),
		I(n, n), x_hat(n)
{
    I.setIdentity();
}

void KalmanFilter::Init(const Eigen::VectorXd& x0) 
{
	x_hat = x0;
	P = P0;
	initialized = true;
}

void KalmanFilter::Init() 
{
	x_hat.setZero();
	P = P0;
	initialized = true;
}

void KalmanFilter::Set_matrix(
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

void KalmanFilter::Predict(const Eigen::VectorXd& u) 
{
	if(!initialized) {
		std::cout << "Filter is not initialized! Initializing with trivial state.";
		Init();
	}

	x_hat = A*x_hat + B*u;
	P = A*P*A.transpose() + Q;
}

void KalmanFilter::Update(const Eigen::VectorXd& y, const Eigen::MatrixXd& R) 
{
	K = P*H.transpose()*(H*P*H.transpose() + R).inverse();	
	x_hat += K * (y - H*x_hat);

	P = (I - K*H)*P;	
}

void KalmanFilter::Update_dynamics(const Eigen::MatrixXd A) 
{
	this->A = A;
}

void KalmanFilter::Update_output(const Eigen::MatrixXd H) 
{
	this->H = H;
}
