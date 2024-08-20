#pragma once
#include "matrix.hpp"
#include "vector.hpp"
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

void run(){
    testMatrixDefaultCtor();
    testMatrixDefaultCtorNonSquare();
    testMatrixAt();
    testMatrixHostSGEMMSquare();
    testMatrixHostSGEMMRec();
    testMatrixDeviceSGEMMSquare();
    testMatrixDeviceSGEMMRec();
	testVectorDefaultCtor();
}
