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