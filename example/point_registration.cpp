#include "Engine.h"
#include <fstream>
#include <iostream>
#include "omp.h"

void create_data(Eigen::MatrixXd &pa, Eigen::MatrixXd &pb) {
	int n_data = 7;
   	Eigen::MatrixXd pa0(3, n_data);	
   	Eigen::MatrixXd pb0(3, n_data);	
	
	//pa0 << 0.31259587832859587,0.18183369508811406,-0.1985829451254436,0.30956936308315824,0.18325744143554143,0.27386997853006634,0.31092539855412077,-0.24102318712643214,-0.19934268082891193,0.3083037180559976,-0.24369886943272182,0.26373498354639324,-0.41887309721537996,0.18325787356921605,-0.20897817398820606,-0.41978328994342257,0.18285575083323888,0.2791437378951481,-0.4027379708630698,-0.24648270436695643,-0.2098449000290462; 
	pb0 << -0.7045189014502934,0.31652495664145264,-0.8913587885243552,0.4196143278053829,0.33125081405575785,-1.148712511573519,-0.7211957446166447,-0.4204243223315903,-0.8922857301575797,0.41556308950696674,-0.36760757371251074,-1.1630155401570457,-0.12535642300333297,0.26708755761917147,1.5095344824450356,0.9968448409012386,0.27593113974268946,1.2189108175890786,-0.28095118914331707,-0.40276257201497045,1.3669272703783852; 
	
	//pa = pA1.transpose();
	Eigen::Quaterniond q(2.86073, 0.0378363, 3.59752, 0.4211619);
	q.normalize();
	std::cout << "q00: x " << q.x() << " y: " << q.y() << " z " << q.z() << " w " << q.w() << std::endl;
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
	CMAES::cost_type fcost = [&](const double *params, int n_params) {
		Eigen::Matrix3d rotation = Eigen::Quaterniond(params[0], params[1], params[2], params[3]).toRotationMatrix();
		
		//std::cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		Eigen::Matrix3d scale = Eigen::Vector3d(params[4], params[5], params[6]).cwiseAbs().asDiagonal();
		
		//std::cout << scale << std::endl;
		Eigen::MatrixXd y = rotation*scale*pa;
		double cost = (pb - y).squaredNorm();
		return cost;
    };

	CMAES::transform_type ftransform = [&](double *params, int n_params) {
		Eigen::Quaterniond q(params[0], params[1], params[2], params[3]); 
		q.normalize();
		params[0] = q.x();
		params[1] = q.y();
		params[2] = q.z();
		params[3] = q.w();
	};
    // <-

	
	//double sigma = 1;
  	//CMAParameters<> cmaparams(x0std, sigma);
  	//CMASolutions cmasols = cmaes<>(fcost, cmaparams);
  	//std::cout << "best solution: " << cmasols << std::endl;
  	//std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds\n";
	
	CMAES::Engine cmaes;
    Eigen::VectorXd x0(n_params);
	//x0 << q00.x(), q00.y(), q00.z(), q00.w(), scale00(0,0), scale00(1,1), scale00(2,2);
	x0 << 0, 0, 0, 1, 1, 1, 1;
  	double c = fcost(x0.data(), n_params);
	std::cout << "x0: " << x0.transpose() << " fcost(x0): " << c << std::endl;
    double sigma0 = 1;
    Solution sol = cmaes.fmin(x0, n_params, sigma0, 6, 999, fcost, ftransform);
	
	std::cout << "\nf_best: " << sol.f << "\nparams_best: " << sol.params.transpose() << std::endl;

    return 0;
}
