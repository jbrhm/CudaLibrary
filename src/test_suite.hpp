#pragma once
#include "matrix/matrix.hpp"
#include "vector/vector.hpp"
#include <iostream>
#include <exception>
#include <sstream>

template<typename T1, typename T2>
void ASSERT_EQUAL(T1 a, T2 b){
    if(a != b){
        throw std::runtime_error("Assertion Failed");
    }
}

void ASSERT_TRUE(bool clause){
    if(!clause){
        throw std::runtime_error("Assertion Failed");
    }
}

void testMatrixDefaultCtor(){
    unsigned int m = 3;
    unsigned int n = 3;
    Matrix m1(m,n);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m1.at(row, col), 1);
            }else{
                ASSERT_EQUAL(m1.at(row, col), 0);
            }
        }
    }
}

void testMatrixDefaultCtorNonSquare(){
    unsigned int m = 3;
    unsigned int n = 10;
    Matrix m1(m,n);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m1.at(row, col), 1);
            }else{
                ASSERT_EQUAL(m1.at(row, col), 0);
            }
        }
    }
}

void testMatrixAt(){
    // Regular at
    unsigned int m = 3;
    unsigned int n = 10;
    Matrix m1(m,n);
    m1.at(2,9) = 2;

    ASSERT_EQUAL(m1.at(2,9), 2);

    // Const at
    float data[]{1, 2, 3};
    Matrix const m2(1,3, data);

    ASSERT_EQUAL(m2.at(0, 2), 3);
}

void testMatrixHostSGEMMSquare(){
    unsigned int m = 3;
    unsigned int n = 3;
    unsigned int k = 3;

    Matrix m1(m, k);
    float data[]{2, 0, 0, 0, 2, 0, 0, 0, 2};
    Matrix m2(k, n, data);

    Matrix m3(m, n);

    Matrix::mySGEMM(m1, m2, m3);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m3.at(row, col), 2);
            }else{
                ASSERT_EQUAL(m3.at(row, col), 0);
            }
        }
    }
}

void testMatrixHostSGEMMRec(){
    unsigned int m = 4;
    unsigned int n = 3;
    unsigned int k = 3;

    Matrix m1(m, k);
    float data[]{2, 0, 0, 0, 2, 0, 0, 0, 2};
    Matrix m2(k, n, data);

    Matrix m3(m, n);

    Matrix::mySGEMM(m1, m2, m3);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m3.at(row, col), 2);
            }else{
                ASSERT_EQUAL(m3.at(row, col), 0);
            }
        }
    }
}

void testMatrixDeviceSGEMMSquare(){
    unsigned int m = 1000;
    unsigned int n = 1000;
    unsigned int k = 1000;

    Matrix m1(m, k);
    Matrix m2(k, n);

    for(unsigned int row = 0; row < k; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                m2.at(row, col) = 2;
            }
        }
    }

    Matrix m3(m, n);

    Matrix::mySGEMM(m1, m2, m3);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m3.at(row, col), 2);
            }else{
                ASSERT_EQUAL(m3.at(row, col), 0);
            }
        }
    }
}

void testMatrixDeviceSGEMMRec(){
    unsigned int m = 1001;
    unsigned int n = 1000;
    unsigned int k = 1000;

    Matrix m1(m, k);
    Matrix m2(k, n);

    for(unsigned int row = 0; row < k; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                m2.at(row, col) = 2;
            }
        }
    }

    Matrix m3(m, n);

    Matrix::mySGEMM(m1, m2, m3);

    for(unsigned int row = 0; row < m; ++row){
        for(unsigned int col = 0; col < n; ++col){
            if(row == col){
                ASSERT_EQUAL(m3.at(row, col), 2);
            }else{
                ASSERT_EQUAL(m3.at(row, col), 0);
            }
        }
    }
}

void testVectorDefaultCtor(){
	Vector v{4};
	std::istringstream correct("[ 0 0 0 0 ]");
	std::ostringstream result;
	v.print(result);

	ASSERT_TRUE(correct.str() == result.str());
}

void testVectorDataCtor(){
	std::vector<float> data{1, 3, 4, 4, 4, 5};
	Vector v{static_cast<unsigned int>(data.size()), data.data()};
	std::istringstream correct("[ 1 3 4 4 4 5 ]");
	std::ostringstream result;
	v.print(result);

	ASSERT_TRUE(correct.str() == result.str());
}

void testVectorAdd(){
	std::vector<float> data(257, 1);
	Vector v1{static_cast<unsigned int>(data.size()), data.data()};
	Vector v2{static_cast<unsigned int>(data.size()), data.data()};
	Vector v3{static_cast<unsigned int>(data.size()), data.data()};

	std::istringstream correct("[ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ]");
	std::ostringstream result;

	Vector::vectorAdd(v1, v2, v3);

	v3.print(result);

	ASSERT_TRUE(correct.str() == result.str());
}

void testVectorAVXAdd(){
	std::vector<float> data{1, 3, 4, 4, 4, 5};
	std::vector<float> data2{1, 3, 4, 4, 4, 5};
	Vector v1{static_cast<unsigned int>(data.size()), data.data()};
	Vector v2{static_cast<unsigned int>(data2.size()), data2.data()};

	Vector out{static_cast<unsigned int>(data.size()), data.data()};

	v1.syncAVX();
	v2.syncAVX();
	out.syncAVX();

	Vector::vectorAdd(v1, v2, out);

	out.syncHost();

	std::istringstream correct("[ 2 6 8 8 8 10 ]");
	std::ostringstream result;
	out.print(result);

	ASSERT_TRUE(correct.str() == result.str());
}

void testVectorHostAdd(){
	std::vector<float> data{1, 3, 4, 4, 4, 5};
	Vector v1{static_cast<unsigned int>(data.size()), data.data()};
	Vector v2{static_cast<unsigned int>(data.size()), data.data()};

	Vector out{static_cast<unsigned int>(data.size()), data.data()};

	Vector::vectorAdd(v1, v2, out);

	std::istringstream correct("[ 2 6 8 8 8 10 ]");
	std::ostringstream result;
	out.print(result);

	ASSERT_TRUE(correct.str() == result.str());
}

void run(){
    //testMatrixDefaultCtor();
    //testMatrixDefaultCtorNonSquare();
    //testMatrixAt();
    //testMatrixHostSGEMMSquare();
    //testMatrixHostSGEMMRec();
    //testMatrixDeviceSGEMMSquare();
    //testMatrixDeviceSGEMMRec();
	//testVectorDefaultCtor();
	//testVectorDataCtor();
	testVectorAdd();
	//testVectorAVXAdd();
	//testVectorHostAdd();
}
