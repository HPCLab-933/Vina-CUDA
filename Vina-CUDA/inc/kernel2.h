#pragma once

#include <stdexcept>
#include <cuda_fp16.h>

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			   static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		throw std::runtime_error("CUDA Runtime Error");
	}
}
#define checkCUDA(val) check((val), #val, __FILE__, __LINE__)

typedef float flo;
// typedef half flo;

// Macros below are shared in both device and host
#define TOLERANCE 1e-16
// kernel1 macros
#define MAX_NUM_OF_EVERY_M_DATA_ELEMENT 512
#define MAX_M_DATA_MI 16
#define MAX_M_DATA_MJ 16
#define MAX_M_DATA_MK 16
#define MAX_NUM_OF_TOTAL_M_DATA MAX_M_DATA_MI *MAX_M_DATA_MJ *MAX_M_DATA_MK *MAX_NUM_OF_EVERY_M_DATA_ELEMENT

// kernel2 macros
#define MAX_NUM_OF_DIMENSIONS_OF_MATIRX (3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1)
#define MAX_NUM_OF_LIG_TORSION 48
#define MAX_NUM_OF_FLEX_TORSION 1
#define MAX_NUM_OF_RIGID 48
#define MAX_NUM_OF_ATOMS 128
#define SIZE_OF_MOLEC_STRUC ((3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) * sizeof(flo))
#define SIZE_OF_CHANGE_STRUC ((3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) * sizeof(flo))
#define MAX_HESSIAN_MATRIX_SIZE ((6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION) * (6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) / 2)
#define MAX_NUM_OF_LIG_PAIRS 2048 // original 4096
#define MAX_NUM_OF_BFGS_STEPS 64
#define MAX_NUM_OF_RANDOM_MAP 20000 // 20000 // not too large (stack overflow!)
#define MAX_NUM_OF_RANDOM_MAP_ 1000
#define GRIDS_SIZE 17
#define MAX_NUM_OF_PROTEIN_ATOMS 50000

#ifdef LARGE_BOX
// docking box size <= 100x100x100
#define MAX_NUM_OF_GRID_MI 300
#define MAX_NUM_OF_GRID_MJ 300
#define MAX_NUM_OF_GRID_MK 300
#define MAX_NUM_OF_ATOM_RELATION_COUNT 38000
#endif

#ifdef MEDIUM_BOX
// docking box size <= 70x70x70
#define MAX_NUM_OF_GRID_MI 200
#define MAX_NUM_OF_GRID_MJ 200
#define MAX_NUM_OF_GRID_MK 200
#define MAX_NUM_OF_ATOM_RELATION_COUNT 20000
#endif

#ifdef SMALL_BOX
// docking box size <= 30x30x30
#define MAX_NUM_OF_GRID_MI 128
#define MAX_NUM_OF_GRID_MJ 128
#define MAX_NUM_OF_GRID_MK 128
#define MAX_NUM_OF_ATOM_RELATION_COUNT 1024
#endif

// #define GRID_MI 65//55
// #define GRID_MJ 71//55
// #define GRID_MK 61//81
#define MAX_P_DATA_M_DATA_SIZE 256
// #define MAX_NUM_OF_GRID_ATOMS 130
#define FAST_SIZE 2051
#define SMOOTH_SIZE 2051
#define MAX_CONTAINER_SIZE_EVERY_WI 5
// #define EL_TYPE_H_CL 0
#define EL_TYPE_H 0

#define EL_TYPE_SIZE 11
#define AD_TYPE_SIZE 20
#define XS_TYPE_SIZE 17
#define SY_TYPE_SIZE 18

typedef struct
{
	flo data[GRIDS_SIZE];
} affinities_cl;

typedef struct
{
	int types[4]; // el ad xs sy
	flo coords[3];
} atom_cl;

typedef struct
{
	atom_cl atoms[MAX_NUM_OF_PROTEIN_ATOMS];
} pa_cl;

typedef struct
{
	flo coords[MAX_NUM_OF_ATOMS][3];
} m_coords_cl;

typedef struct
{
	flo coords[MAX_NUM_OF_ATOMS][3];
} ligand_atom_coords_cl;

typedef struct
{
	flo coords[MAX_NUM_OF_ATOMS][3];
	// int lock[MAX_NUM_OF_ATOMS];
} m_minus_forces;

typedef struct
{ // namely molec_struc
	flo e;
	flo position[3];
	flo orientation[4];
	flo lig_torsion[MAX_NUM_OF_LIG_TORSION];
	flo flex_torsion[MAX_NUM_OF_FLEX_TORSION];
	// flo coords		[MAX_NUM_OF_ATOMS][3];
	// flo lig_torsion_size;
} output_type_cl;

typedef struct
{ // namely change_struc
	// flo lig_torsion_size;
	flo position[3];
	flo orientation[3];
	flo lig_torsion[MAX_NUM_OF_LIG_TORSION];
	flo flex_torsion[MAX_NUM_OF_FLEX_TORSION];
} change_cl;

typedef struct
{ // depth-first order
	int num_children;
	bool children_map[MAX_NUM_OF_RIGID][MAX_NUM_OF_RIGID]; // chidren_map[i][j] = true if node i's child is node j
	int parent[MAX_NUM_OF_RIGID];						   // every node has only 1 parent node

	int atom_range[MAX_NUM_OF_RIGID][2];
	flo origin[MAX_NUM_OF_RIGID][3];
	flo orientation_m[MAX_NUM_OF_RIGID][9]; // This matrix is fixed to 3*3
	flo orientation_q[MAX_NUM_OF_RIGID][4];

	flo axis[MAX_NUM_OF_RIGID][3];			  // 1st column is root node, all 0s
	flo relative_axis[MAX_NUM_OF_RIGID][3];	  // 1st column is root node, all 0s
	flo relative_origin[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s

} rigid_cl;

typedef struct
{
	int num_pairs;
	int type_pair_index[MAX_NUM_OF_LIG_PAIRS];
	int a[MAX_NUM_OF_LIG_PAIRS];
	int b[MAX_NUM_OF_LIG_PAIRS];
} lig_pairs_cl;

typedef struct
{
	int begin;
	int end;
	lig_pairs_cl pairs;
	rigid_cl rigid;
} ligand_cl;

typedef struct
{
	int begin;
	int end;
	int m_num_movable_atoms;
	// lig_pairs_cl pairs;
	rigid_cl rigid;
} ligand_gpu;

typedef struct
{
	int int_map[MAX_NUM_OF_RANDOM_MAP];
	flo pi_map[MAX_NUM_OF_RANDOM_MAP];
	flo sphere_map[MAX_NUM_OF_RANDOM_MAP][3];
} random_maps;

typedef struct
{
	int m_num_movable_atoms;
	atom_cl atoms[MAX_NUM_OF_ATOMS];
	m_coords_cl m_coords;		 // flo coords[128][3];
	m_minus_forces minus_forces; // flo coords[128][3];
	ligand_cl ligand;
} m_cl;

typedef struct
{
	int m_i;
	int m_j;
	int m_k;
	flo m_init[3];
	flo m_range[3];
	flo m_factor[3];
	flo m_dim_fl_minus_1[3];
	flo m_factor_inv[3];
	flo m_data[(MAX_NUM_OF_GRID_MI) * (MAX_NUM_OF_GRID_MJ) * (MAX_NUM_OF_GRID_MK) * 8];
} grid_cl;

typedef struct
{
	int atu; // atom_typing_used
	flo slope;
	grid_cl grids[GRIDS_SIZE];
} grids_cl;

typedef struct
{
	flo factor;
	flo fast[FAST_SIZE];
	flo smooth[SMOOTH_SIZE][2];
} p_m_data_cl;

typedef struct
{
	int n;
	flo m_cutoff_sqr;
	flo factor;
	p_m_data_cl m_data[MAX_P_DATA_M_DATA_SIZE];
} pre_cl;

typedef struct
{
	int dims[3];
	flo init[3];
	flo range[3];
} gb_cl;

typedef struct
{
	int relation[MAX_NUM_OF_ATOM_RELATION_COUNT][MAX_NUM_OF_ATOM_RELATION_COUNT];
	int relation_size[MAX_NUM_OF_ATOM_RELATION_COUNT];
} ar_cl;

typedef struct
{
	int needed_size;
	// int torsion_size;
	// int search_depth;
	// int max_bfgs_steps;
	int total_wi;
	int thread;
	int ar_mi;
	int ar_mj;
	int ar_mk;
	int grids_front;

	flo epsilon_fl;
	flo cutoff_sqr;
	flo max_fl;
	flo mutation_amplitude;
	flo hunt_cap[3];
	flo authentic_v[3];
} mis_cl;

typedef struct
{
	int max_steps;
	flo average_required_improvement;
	int over;
	int ig_grids_m_data_step;
	int p_data_m_data_step;
	int atu;
	int m_num_movable_atoms;
	flo slope;
	flo epsilon_fl;
	flo epsilon_fl2;
	flo epsilon_fl3;
	flo epsilon_fl4;
	flo epsilon_fl5;
} variables_bfgs;

typedef struct
{
	output_type_cl container[MAX_CONTAINER_SIZE_EVERY_WI];
	int current_size;
} out_container;

typedef struct
{
	atom_cl m_gpu_atom[MAX_NUM_OF_ATOMS];
} m_atom_gpu;

typedef struct
{
	// int m_i;
	// int m_j;
	// int m_k;
	// flo m_init[3];
	// flo m_range[3];
	// flo m_factor[3];
	// flo m_dim_fl_minus_1[3];
	// flo m_factor_inv[3];
	flo m_data[(MAX_NUM_OF_GRID_MI) * (MAX_NUM_OF_GRID_MJ) * (MAX_NUM_OF_GRID_MK) * 8];
} grid_cl_m;

typedef struct
{
	int m_i;
	int m_j;
	int m_k;
	flo m_init[3];
	flo m_range[3];
	flo m_factor[3];
	flo m_dim_fl_minus_1[3];
	flo m_factor_inv[3];
} grids_cl_other;

typedef struct
{
	grids_cl_other grid_other[GRIDS_SIZE];
} grids_gpu;

typedef struct
{
	int atu;
	flo slope;
	grid_cl_m grids_m[GRIDS_SIZE];
} grids_gpu_m;
