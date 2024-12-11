/*
__device__ __forceinline__ void get_heavy_atom_movable_coords(output_type_cl* tmp, const m_cl* m, ligand_atom_coords_cl* coords);

__device__ __forceinline__  flo generate_n( flo* pi_map, const int step); // // delete the constant 

__device__ __forceinline__ bool metropolis_accept(flo old_f, flo new_f, flo temperature, flo n);

__global__  void kernel2(
    const				output_type_cl*			ric,  // delete the __global
                        m_cl*					mg, // delete the __global
                        pre_cl*					pre, // delete the __constant
                        grids_cl*				grids, // delete the __constant
                        random_maps*			random_maps, // delete the __constant
                        ligand_atom_coords_cl*	coords, // delete the __global
                        output_type_cl*			results, // delete the __global
    const				mis_cl*					mis, // delete the __global
    const				int						torsion_size,
    const				int						search_depth,
    const				int						max_bfgs_steps,
    const 				int						rilc_bfgs_enable
);*/

//__constant__ mis_cl mis_cuda;

extern "C"
void kernel_monte(
				const				output_type_cl*			ric,  // delete the __global
						    		m_cl*					mg, // delete the __global
       				pre_cl*					pre, // delete the __constant
					    grids_cl*				grids, // delete the __constant
				        random_maps*			random_maps, // delete the __constant
							        ligand_atom_coords_cl*	coords, // delete the __global
									output_type_cl*			results, // delete the __global
				const				mis_cl*					mis, // delete the __global
				const				int						torsion_size,
				const				int						search_depth,
				const				int						max_bfgs_steps,
				const 				int						rilc_bfgs_enable,
                // quickvina-W
                float center_x,
                float center_y,
                float center_z,
                float size_x,
                float size_y,
                float size_z,
                ele_cl* global_ptr,
                int* count_id
);