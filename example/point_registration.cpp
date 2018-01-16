#include "Engine.h"
#include <fstream>
#include <iostream>
#include "omp.h"
#include "cmaes.h"

using namespace libcmaes;

void create_data(Eigen::MatrixXd &pA, Eigen::MatrixXd &pB) {
	int n_data = 7;
   	Eigen::MatrixXd pA1(n_data, 3);	
   	Eigen::MatrixXd pB1(n_data, 3);	
	
	pA1 << 0.31259587832859587,0.18183369508811406,-0.1985829451254436,0.30956936308315824,0.18325744143554143,0.27386997853006634,0.31092539855412077,-0.24102318712643214,-0.19934268082891193,0.3083037180559976,-0.24369886943272182,0.26373498354639324,-0.41887309721537996,0.18325787356921605,-0.20897817398820606,-0.41978328994342257,0.18285575083323888,0.2791437378951481,-0.4027379708630698,-0.24648270436695643,-0.2098449000290462; 
	pB1 << -0.7045189014502934,0.31652495664145264,-0.8913587885243552,0.4196143278053829,0.33125081405575785,-1.148712511573519,-0.7211957446166447,-0.4204243223315903,-0.8922857301575797,0.41556308950696674,-0.36760757371251074,-1.1630155401570457,-0.12535642300333297,0.26708755761917147,1.5095344824450356,0.9968448409012386,0.27593113974268946,1.2189108175890786,-0.28095118914331707,-0.40276257201497045,1.3669272703783852; 
	
	pA = pA1.transpose();
	pB = pB1.transpose();
	
}


int main() {
	
    //-> create and fill synthetic data
	int n_params = 7;
    Eigen::VectorXd u(n_params);
    Eigen::MatrixXd pA, pB;
    create_data(pA, pB);
    // <-

	   
	Eigen::Quaterniond q00(2.86073, 0.0378363, 3.59752, 0.0211619);	
	q00.normalize();
	Eigen::Matrix3d rotation00 = q00.toRotationMatrix();
	Eigen::Matrix3d scale00 = Eigen::Vector3d(3.31978, 1.64983, 2.48582).asDiagonal();
	
	std::cout << rotation00 << std::endl; 

    // -> cost function
    FitFunc fcost = [&](const double *params, const int n_params) {
		Eigen::Quaterniond q(params[0], params[1], params[2], params[3]); 
		//std::cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		q.normalize();
		Eigen::Matrix3d rotation = q.toRotationMatrix();
		
		//std::cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		Eigen::Matrix3d scale1 = Eigen::Vector3d(params[4], params[5], params[6]).asDiagonal();
		Eigen::Matrix3d scale = scale1.cwiseAbs();	
		
		//std::cout << scale << std::endl;
		Eigen::MatrixXd y = rotation*scale*pA;
		double cost = (pB - y).squaredNorm();
		return cost;
    };
    // <-

		
    Eigen::VectorXd x0(n_params);
	x0 << 1, 0, 0, 0, 1, 1, 1;
	//x0 << q.w(), q.x(), q0.y(), 0, 1, 1, 1;
	//x0 << 0, 1, 2, 3, 4, 5, 6;
	std::vector<double> x0std(x0.data(), x0.data() + n_params);
  	double c = fcost(x0std.data(), n_params);
	std::cout << "cost0: " << c << std::endl;

	
	double sigma = 1;
  	CMAParameters<> cmaparams(x0std, sigma);
  	CMASolutions cmasols = cmaes<>(fcost, cmaparams);
  	std::cout << "best solution: " << cmasols << std::endl;
  	std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds\n";
	
	//CMAES::Engine cmaes;
    //Eigen::VectorXd x0(n_params);
	//x0 << q00.x(), q00.y(), q00.z(), q00.w(), scale00(0,0), scale00(1,1), scale00(2,2);
	//x0 << 0, 0, 0, 1, 1, 1, 1;
	//double c = cost_func(x0, x0, n_params);
	//std::cout << "x0: " << c << std::endl;
    //double sigma0 = 1;
    //Solution sol = cmaes.fmin(x0, n_params, sigma0, 6, 999, cost_func, transform_scale_shift);

    return 0;
}
