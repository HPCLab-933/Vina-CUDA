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
				const 				int						rilc_bfgs_enable
);