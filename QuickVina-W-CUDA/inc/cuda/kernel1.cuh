
/*
__device__ __forceinline__ int fl_to_sz(flo x, flo max_sz);
__device__ __forceinline__ int num_atom_types(int atu);
__device__ __forceinline__ const flo vec_distance_sqr(const flo* a, const flo* b);
__device__ __forceinline__ const int triangular_matrix_index(int n, int i, int j);
__device__ __forceinline__ const int triangular_matrix_index_permissive(int n, int i, int j);
__device__ __forceinline__ flo eval_fast(int type_pair_index, flo r2, flo cutoff_sqr, const pre_cl* pre);
__device__ __forceinline__ const int* possibilities(						flo*		coords,
	const						ar_cl*		ar,  	 // delete the __global
	const						flo		epsilon_fl, 
	const						gb_cl*		gb,      // delete the __global
								int*		relation_count,
	const	                	mis_cl*		mis      // delete the __global
);

__global__ void kernel1(
	const				pre_cl*		pre, 
	const				pa_cl*		pa,  
	const				gb_cl*		gb,  
	const				ar_cl*		ar,  
						grids_cl*	grids, 
	const				mis_cl*		mis,   
	const				flo*		needed,
	const				int			atu,
	const				int			nat
);
*/

extern "C"
void kernel_grid(
				const				pre_cl*		pre, // delete the __global
				const				pa_cl*		pa,  // delete the __global
				const				gb_cl*		gb,  // delete the __global
				const				ar_cl*		ar,  // delete the __global
									grids_cl*	grids, // delete the __global
				const				mis_cl*		mis,   // delete the __global
				const				flo*		needed,// delete the __global
				const				int			atu,
				const				int			nat
);