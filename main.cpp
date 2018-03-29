#include <iostream>
#include <ctime>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#define Matrix_SIZE 100

int main( int argc, char** argv )
{

    //解方程
    Eigen::Matrix < double, Matrix_SIZE, Matrix_SIZE > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( Matrix_SIZE, Matrix_SIZE );
    Eigen::Matrix< double, Matrix_SIZE, 1 > v_Nd;
    v_Nd = Eigen::MatrixXd::Random( Matrix_SIZE, 1 );

    clock_t time_stt = clock(); //计时

    //求逆
    Eigen::Matrix< double , Matrix_SIZE ,1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " <<1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms" << endl;

    //QR分解
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    //cout << "The x is " << endl << x << endl;
    cout << "time use in QR composition  is " <<1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms" << endl;

    //cholesky分解
    //Eigen::MatrixXd A(3,3);
    //A << 4,-1,2, -1,6,0, 2,0,5;
    //cout << "The matrix A is" << endl << A << endl;
    Eigen::LLT< Eigen::MatrixXd > lltOfA(matrix_NN); // compute the Cholesky decomposition of A
    Eigen::MatrixXd L = lltOfA.matrixL(); // retrieve factor L  in the decomposition
// The previous two lines can also be written as "L = A.llt().matrixL()"
    cout << "The Cholesky factor L is" << endl << L << endl;
    cout << "To check this, let us compute L * L.transpose()" << endl;
    cout << L * L.transpose() << endl;
    cout << "This should equal the matrix A" << endl;

    return 0;


}