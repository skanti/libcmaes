#include "Engine.h"
#include <fstream>
#include <iostream>
#include "omp.h"

void create_data(Eigen::MatrixXd &pa, Eigen::MatrixXd &pb) {
	int n_data = 7;
   	Eigen::MatrixXd pa0(3, n_data);	
   	Eigen::MatrixXd pb0(3, n_data);	
	
	pb0 << -0.7045189014502934,0.31652495664145264,-0.8913587885243552,0.4196143278053829,0.33125081405575785,-1.148712511573519,-0.7211957446166447,-0.4204243223315903,-0.8922857301575797,0.41556308950696674,-0.36760757371251074,-1.1630155401570457,-0.12535642300333297,0.26708755761917147,1.5095344824450356,0.9968448409012386,0.27593113974268946,1.2189108175890786,-0.28095118914331707,-0.40276257201497045,1.3669272703783852; 
	
	Eigen::Quaterniond q = Eigen::Quaterniond(2.86073, 0.0378363, 3.59752, 0.4211619).normalized();
	std::cout << "groundtruth-quaternion. w: " << q.w() << " x " << q.x() << " y: " << q.y() << " z " << q.z() << std::endl;
	pa = q.toRotationMatrix()*pb0;
	pb = pb0;
}


int main() {
	
    //-> create and fill synthetic data
	int n_params = 7;
    Eigen::VectorXd u(n_params);
    Eigen::MatrixXd pa, pb;
    create_data(pa, pb);
    // <-

    // -> cost function
	CMAES::cost_type fcost = [&](double *params, int n_params) {
		Eigen::Quaterniond q = Eigen::Quaterniond(params[0], params[1], params[2], params[3]).normalized();
		Eigen::Vector3d s = Eigen::Vector3d(params[4], params[5], params[6]).cwiseAbs();

		params[0] = q.w();
		params[1] = q.x();
		params[2] = q.y();
		params[3] = q.z();
		params[4] = s(0);
		params[5] = s(1);
		params[6] = s(2);

		Eigen::Matrix3d rotation = q.toRotationMatrix();
		
		Eigen::Matrix3d scale = s.asDiagonal();
		
		Eigen::MatrixXd y = rotation*scale*pa;
		double cost = (pb - y).squaredNorm();
		return cost;
    };

	
	CMAES::Engine cmaes;
    Eigen::VectorXd x0(n_params);
	x0 << 1, 0, 0, 0, 1, 1, 1;
  	double c = fcost(x0.data(), n_params);
	std::cout << "x0: " << x0.transpose() << " fcost(x0): " << c << std::endl;
    double sigma0 = 1;
    Solution sol = cmaes.fmin(x0, n_params, sigma0, 6, 999, fcost);
	
	std::cout << "\nf_best: " << sol.f << "\nparams_best: " << sol.params.transpose() << std::endl;

    return 0;
}
