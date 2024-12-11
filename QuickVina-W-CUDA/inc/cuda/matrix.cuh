#pragma once

#include "commonMacros.h"

// symmetric matrix (only half of it are stored)
typedef struct {
	flo data[MAX_HESSIAN_MATRIX_SIZE];
	int dim;
}matrix_gpu;

__device__ inline void matrix_init(matrix_gpu* m, int dim, flo fill_data) {
	m->dim = dim;
	if ((dim * (dim + 1) / 2) > MAX_HESSIAN_MATRIX_SIZE)printf("\nnmatrix: matrix_init() ERROR!");
	//((dim * (dim + 1) / 2)*sizeof(flo)); // symmetric matrix
	for (int i = 0; i < (dim * (dim + 1) / 2); i++)m->data[i] = fill_data;
	for (int i = (dim * (dim + 1) / 2); i < MAX_HESSIAN_MATRIX_SIZE; i++)m->data[i] = 0;// Others will be 0
}

// as rugular 3x3 matrix
__device__ inline void mat_init(matrix_gpu* m, flo fill_data) {
	m->dim = 3; // fixed to 3x3 matrix
	if (9 > MAX_HESSIAN_MATRIX_SIZE)printf("\nnmatrix: mat_init() ERROR!");
	for (int i = 0; i < 9; i++)m->data[i] = fill_data;
}


__device__ inline void matrix_set_diagonal(matrix_gpu* m, flo fill_data) {
	for (int i = 0; i < m->dim; i++) {
		m->data[i + i * (i + 1) / 2] = fill_data;
	}
}

// as rugular matrix
__device__ inline void matrix_set_element(matrix_gpu* m, int dim, int x, int y, flo fill_data) {
	m->data[x + y * dim] = fill_data;
}

__device__ inline void matrix_set_element_tri(matrix_gpu* m, int x, int y, flo fill_data) {
	m->data[x + y*(y+1)/2] = fill_data;
}
__device__ inline int tri_index(int n, int i, int j) {
	if (j >= n || i > j)printf("\nmatrix: tri_index ERROR!");
	return i + j * (j + 1) / 2;
}

__device__ inline int index_permissive(const matrix_gpu* m, int i, int j) {
	return (i < j) ? tri_index(m->dim, i, j) : tri_index(m->dim, j, i);
}