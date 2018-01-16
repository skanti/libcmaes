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
	
	//pA1 << 0.31259587832859587,0.18183369508811406,-0.1985829451254436,0.30956936308315824,0.18325744143554143,0.27386997853006634,0.31092539855412077,-0.24102318712643214,-0.19934268082891193,0.3083037180559976,-0.24369886943272182,0.26373498354639324,-0.41887309721537996,0.18325787356921605,-0.20897817398820606,-0.41978328994342257,0.18285575083323888,0.2791437378951481,-0.4027379708630698,-0.24648270436695643,-0.2098449000290462; 
	//pB1 << -0.7045189014502934,0.31652495664145264,-0.8913587885243552,0.4196143278053829,0.33125081405575785,-1.148712511573519,-0.7211957446166447,-0.4204243223315903,-0.8922857301575797,0.41556308950696674,-0.36760757371251074,-1.1630155401570457,-0.12535642300333297,0.26708755761917147,1.5095344824450356,0.9968448409012386,0.27593113974268946,1.2189108175890786,-0.28095118914331707,-0.40276257201497045,1.3669272703783852; 
	pA1 << 0.3684166669845581,0.22149555385112762,-0.233120396733284,0.3653901517391205,0.222919300198555,0.23933252692222595,0.366746187210083,-0.20136132836341858,-0.23388013243675232,0.36412450671195984,-0.20403701066970825,0.22919753193855286,-0.3630523085594177,0.22291973233222961,-0.24351562559604645,-0.3639625012874603,0.22251760959625244,0.24460628628730774,-0.34691718220710754,-0.20682084560394287,-0.2443823516368866;
	pB1 << -0.7024705410003662,0.004155769012868404,-0.4333205819129944,0.42166268825531006,0.018881626427173615,-0.6906743049621582,-0.7191473841667175,-0.7327935099601746,-0.43424752354621887,0.4176114499568939,-0.679976761341095,-0.7049773335456848,-0.12330806255340576,-0.045281630009412766,1.9675726890563965,0.9988932013511658,-0.036438047885894775,1.6769490242004395,-0.2789028286933899,-0.7151317596435547,1.824965476989746;
		
	pA = pA1.transpose();
	pB = pB1.transpose();
	
	pA = pA - pA.rowwise().mean();
	pB = pB - pB.rowwise().mean();

	std::cout << pA << std::endl;
}


int main() {
    //-> create and fill synthetic data
	int n_params = 7;
    Eigen::VectorXd u(n_params);
    Eigen::MatrixXd pA, pB;
    create_data(pA, pB);
    // <-
	
	Eigen::Matrix3d rotation00; 	
	rotation00 << -0.2585821045651603, 0.01754017857459883, 0.965830025073888,
				0.046257337630971965, 0.9989129699443112, -0.005756491321216566,
				-0.9648811086936808, 0.04318819992322194, -0.25911238000806697;
	Eigen::Quaterniond q00(rotation00);	
	Eigen::Matrix3d scale00 = Eigen::Vector3d(2.539367693254164, 1.6371537040405728, 3.28130907453851).asDiagonal();
	
	pA = rotation00*pB;

    // -> cost function
    FitFunc fcost = [&](const double *params, const int n_params) {
		Eigen::Quaterniond q(params[0], params[1], params[2], params[3]); 
		
		q.normalize();
		Eigen::Matrix3d rotation = q.toRotationMatrix();
		Eigen::Matrix3d scale1 = Eigen::Vector3d(params[4], params[5], params[6]).asDiagonal();
		Eigen::Matrix3d scale = scale1.cwiseAbs();	
		
		Eigen::MatrixXd y = rotation*scale*pA;
		double cost = (pB - y).squaredNorm();
		return cost;
    };
    // <-

	
    Eigen::VectorXd x0(n_params);
	x0 << 1, 0, 0, 0, 1, 1, 1;
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
