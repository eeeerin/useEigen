#include <iostream>
#include <ctime>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#define Matrix_SIZE 50

int main( int argc, char** argv )
{
    //The Matrix class
    Eigen::Matrix<float, 2, 3> matrix_23; //声明2×3float矩阵
    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix< double , Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    //输入输出数据
    matrix_23 << 1,2,3,4,5,6;
    cout << matrix_23 << endl;

    //用（）访问矩阵中的元素
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << matrix_23(i,j) << endl;
        }
    }

    v_3d << 3,2,1;
    //矩阵向量相乘，先做显式转换
    Eigen::Matrix < double, 2, 1 > result = matrix_23 . cast<double>() * v_3d;
    cout << result << endl;

    matrix_33 = Eigen::MatrixXd::Random();
    cout << matrix_33 << endl ;

    cout << matrix_33.transpose() << endl;
    cout << matrix_33.sum() << endl;
    cout << matrix_33.trace() << endl;
    cout << 10*matrix_33 << endl;
    cout << matrix_33.inverse() << endl;
    cout << matrix_33.determinant() << endl;

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
    cout << "time use in QR composition inverse is " <<1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms" << endl;

    return 0;


}