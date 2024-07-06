#include "matrix.hpp"

extern "C" {
    Matrix* new_matrix(int M, int N){return new Matrix(M, N)}
}