#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel2.h"
#include "mutate_conf.cuh"
#include "quasi_newton.cuh"
#include "matrix.cuh"
#include "assert.h"

__device__ inline void get_heavy_atom_movable_coords(output_type_cl *tmp, const m_cl *m, ligand_atom_coords_cl *coords)
{
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++)
	{
		if (m->atoms[i].types[0] != EL_TYPE_H)
		{
			for (int j = 0; j < 3; j++)
				coords->coords[counter][j] = m->m_coords.coords[i][j];
			counter++;
		}
		else
		{
			// printf("\n kernel2 --> 12: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	// assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++)
	{
		for (int j = 0; j < 3; j++)
			coords->coords[i][j] = 0;
	}
}

__device__ inline void get_heavy_atom_movable_coords_update(output_type_cl *tmp, const m_cl *m, ligand_atom_coords_cl * coords, m_coords_cl* m_coords)
{
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++)
	{
		if (m->atoms[i].types[0] != EL_TYPE_H)
		{
			for (int j = 0; j < 3; j++)
				coords->coords[counter][j] = m_coords->coords[i][j];
			counter++;
		}
		else
		{
			// printf("\n kernel2 --> 12: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	// assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++)
	{
		for (int j = 0; j < 3; j++)
			coords->coords[i][j] = 0;
	}
}

// Generate a random number according to step
__device__ inline flo generate_n(flo *pi_map, const int step)
{ // // delete the constant
	return fabs(pi_map[step]) / M_PI;
}

__device__ inline bool metropolis_accept(flo old_f, flo new_f, flo temperature, flo n)
{
	if (new_f < old_f)
		return true;
	const flo acceptance_probability = exp((old_f - new_f) / temperature);
	bool res = n < acceptance_probability;
	// return n < acceptance_probability; lcf-debug // why not return the res ????
	return res;
}

//__device__ flo m_cutoff_sqr_r =  555.000;
//__device__ flo m_cutoff_sqr_rr =  0;
//__device__ int n = 11;
//__device__ int m = 22;
//__device__ int q = 33;
//__device__ int t = 44;

__global__ void kernel2(
	const output_type_cl *ric,	   // delete the __global
	m_cl *mg,					   // delete the __global
	pre_cl *pre,				   // delete the __constant
	grids_cl *grids,			   // delete the __constant
	random_maps *random_maps,	   // delete the __constant
	ligand_atom_coords_cl *coords, // delete the __global
	output_type_cl *results,	   // delete the __global
	const mis_cl *mis,			   // delete the __global
	const int torsion_size,
	const int search_depth,
	const int max_bfgs_steps,
	const int rilc_bfgs_enable)
{

	// printf("stop the program running"); assert(false);

	// m_cutoff_sqr_rr =  pre->m_cutoff_sqr;
	// printf("LCF-DEBUG the two pre->m_cutoff_sqrr: %f \n",m_cutoff_sqr_rr);
	// for (int i = 0; i < 50; i++) {
	//	n = mg->atoms[i].types[0];
	//	printf("m_ptr->atoms[%d].types[0] = %d \n", i, n);
	// }

	// printf("m_ptr->atoms[i].types[0] = %d \n",n);
	// printf("m_ptr->atoms[i].types[1] = %d \n",m);
	// printf("m_ptr->atoms[i].types[2] = %d \n",q);
	// printf("m_ptr->atoms[i].types[3] = %d \n",t);

	// for (int i = 0; i < 3; i++) {
	//	printf("m_ptr->atoms[i].types[0] = %d \n",&mg->atoms[i].types[0]);
	//	printf("m_ptr->atoms[i].types[1] = %d \n",&mg->atoms[i].types[1]);
	//	printf("m_ptr->atoms[i].types[2] = %d \n",&mg->atoms[i].types[2]);
	//	printf("m_ptr->atoms[i].types[3] = %d \n",&mg->atoms[i].types[3]);
	// }
	/* 锟杰癸拷锟斤拷一锟斤拷block锟斤拷锟斤拷泄锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷停锟�
		1.output_type_cl* ric,


	*/
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gs = gridDim.x * blockDim.x;
	int gl = gx + gy * gs;

	// printf("the gl is : %d \n,", gl);

	flo best_e = INFINITY;
	output_type_cl best_out;
	ligand_atom_coords_cl best_coords;

	for (int gll = gl; gll < mis->thread; gll += mis->total_wi)
	{
		m_cl m = *mg;

		output_type_cl tmp = ric[gll];

		change_cl g;

		output_type_cl candidate;

		for (int step = 0; step < search_depth; step++)
		{
			candidate = tmp;

			int map_index = (step + gll * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			mutate_conf_cl(map_index,
						   &candidate,
						   random_maps->int_map,
						   random_maps->sphere_map,
						   random_maps->pi_map,
						   m.ligand.begin,
						   m.ligand.end,
						   m.atoms,
						   &m.m_coords,
						   m.ligand.rigid.origin[0],
						   mis->epsilon_fl,
						   mis->mutation_amplitude,
						   torsion_size);
			// printf(" stop the program running"); assert(true);
			__syncthreads();
			if (rilc_bfgs_enable == 1)
			{
				rilc_bfgs(&candidate,
						  &g,
						  &m,
						  pre,
						  grids,
						  mis,
						  torsion_size,
						  max_bfgs_steps);
			}
			else
			{
				bfgs(&candidate,
					 &g,
					 &m,
					 pre,
					 grids,
					 mis,
					 torsion_size,
					 max_bfgs_steps);
			}

			flo n = generate_n(random_maps->pi_map, map_index);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n))
			{

				tmp = candidate;

				set(&tmp, &m.ligand.rigid, &m.m_coords,
					m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

				if (tmp.e < best_e)
				{
					if (rilc_bfgs_enable == 1)
					{
						rilc_bfgs(&tmp,
								  &g,
								  &m,
								  pre,
								  grids,
								  mis,
								  torsion_size,
								  max_bfgs_steps);
					}
					else
					{
						bfgs(&tmp,
							 &g,
							 &m,
							 pre,
							 grids,
							 mis,
							 torsion_size,
							 max_bfgs_steps);
					}

					// set
					if (tmp.e < best_e)
					{
						set(&tmp, &m.ligand.rigid, &m.m_coords,
							m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

						best_out = tmp;
						// printf("best_out -->> e = %f \n",best_out.e);

						// printf("best_out -->> position = %f , %f, %f \n", best_out.position[0], best_out.position[1], best_out.position[2]);

						// printf("best_out -->> orientation = %f , %f, %f,%f \n", best_out.orientation[0], best_out.orientation[1], best_out.orientation[2],best_out.orientation[3]);
						get_heavy_atom_movable_coords(&best_out, &m, &best_coords); // get coords
						best_e = tmp.e;
					}
				}
			}
		}
		// write the best conformation back to CPU
		results[gll] = best_out;
		coords[gll] = best_coords;
	}
}


__global__ void kernel2_update(
	const output_type_cl *ric,
	m_cl *mg,
	pre_cl *pre,
	grids_cl *grids,
	random_maps *random_maps,
	ligand_atom_coords_cl *coords,
	output_type_cl *results,
	const mis_cl *mis,
	const int torsion_size,
	const int search_depth,
	const int max_bfgs_steps,
	const int rilc_bfgs_enable)
{

	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y +  blockIdx.x;
	// int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;        

	int id = blockNumInGrid;
	// int id = globalThreadNum;

	flo best_e = INFINITY;
	ligand_atom_coords_cl best_coords;

	m_cl* m_gpu = &mg[id];

	__shared__ m_coords_cl m_coords;
	m_coords = m_gpu->m_coords;

	__shared__ m_minus_forces minus_forces;
	minus_forces = m_gpu->minus_forces;

	__shared__ output_type_cl tmp;
	tmp = ric[id];

	__shared__ change_cl g;

	__shared__ output_type_cl best_out;
	__shared__ output_type_cl candidate;

	//__shared__ rigid_cl m_gpu_rigid; // lcf add new
	//m_gpu_rigid = m_gpu->ligand.rigid;   // lcf add new

	__shared__ ligand_gpu m_gpu_ligand;
	m_gpu_ligand.rigid = m_gpu->ligand.rigid;
	m_gpu_ligand.begin =  m_gpu->ligand.begin;
	m_gpu_ligand.end = m_gpu->ligand.end;
	m_gpu_ligand.m_num_movable_atoms = m_gpu->m_num_movable_atoms;
	
	//__shared__ m_atom_gpu atom_gpu;
	//for(int i = threadNumInBlock; i<MAX_NUM_OF_ATOMS; i+=threadsPerBlock){
	//	atom_gpu.m_gpu_atom[i] = m_gpu->atoms[i];
	//}
	//__syncthreads();

	__shared__ mis_cl mis_gpu;
	mis_gpu = *mis;

	__shared__ grids_gpu grids_cuda;

	for(int i = threadNumInBlock; i<GRIDS_SIZE; i+=threadsPerBlock){
		grids_cuda.grid_other[i].m_i = grids->grids[i].m_i;
		grids_cuda.grid_other[i].m_j= grids->grids[i].m_j;
		grids_cuda.grid_other[i].m_k = grids->grids[i].m_k;
		for(int n= 0; n<3;n++){
			grids_cuda.grid_other[i].m_init[n] = grids->grids[i].m_init[n];
			grids_cuda.grid_other[i].m_range[n]  = grids->grids[i].m_range[n] ;
			grids_cuda.grid_other[i].m_factor[n]  = grids->grids[i].m_factor[n] ;
			grids_cuda.grid_other[i].m_dim_fl_minus_1[n]  = grids->grids[i].m_dim_fl_minus_1[n] ;
			grids_cuda.grid_other[i].m_factor_inv[n]  = grids->grids[i].m_factor_inv[n] ;
		}
	}
	__syncthreads();
	

	for (int step = 0; step < search_depth; step++)
	{
		candidate = tmp;
	

		int map_index = (step + id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
		// printf("\nstep = %d, id = %d, search_depth = %d, map_index = %d,", step, id, search_depth, map_index);
		mutate_conf_cl_update(map_index,
							  &candidate,
							  random_maps->int_map,
							  random_maps->sphere_map,
							  random_maps->pi_map,
							  m_gpu_ligand.begin,
							  m_gpu_ligand.end,
							  m_gpu->atoms,
							  &m_coords,
							  m_gpu->ligand.rigid.origin[0],
							  mis_gpu.epsilon_fl,
							  mis_gpu.mutation_amplitude,
							  torsion_size,
							  threadNumInBlock,
							  threadsPerBlock);

		if (rilc_bfgs_enable == 1)
		{
			rilc_bfgs_update(&candidate, // shared memory
							 &g,		 // shared memory
							 m_gpu,	 // global memory
							 pre,
							 grids,
							 &mis_gpu,
							 torsion_size,
							 max_bfgs_steps,
							 &m_coords,		// shared memory
							 &minus_forces, // shared memory
							 threadNumInBlock,
							 threadsPerBlock,
							 &m_gpu_ligand.rigid,
							 &grids_cuda);
		}
		else
		{
			//bfgs(&candidate, // shared memory
			//	 &g,		 // shared memory
			//	 m_gpu,	 // global memory
			//	 pre,
			//	 grids,
			//	 mis,
			//	 torsion_size,
			//	 max_bfgs_steps);
		}

		flo n = generate_n(random_maps->pi_map, map_index);

		if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n))
		{

			tmp = candidate;

			set_update(&tmp, &m_gpu_ligand.rigid, &m_coords,
				m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);

			if (tmp.e < best_e)
			{
				if (rilc_bfgs_enable == 1)
				{
					rilc_bfgs_update(&tmp,
									 &g,
									 m_gpu,
									 pre,
									 grids,
									 &mis_gpu,
									 torsion_size,
									 max_bfgs_steps,
									 &m_coords,		// shared memory
									 &minus_forces, // shared memory
									 threadNumInBlock,
									 threadsPerBlock,
									 &m_gpu_ligand.rigid,
									 &grids_cuda);
				}
				else
				{
					//bfgs(&tmp,
					//	 &g,
					//	 m_gpu,
					//	 pre,
					//	 grids,
					//	 mis,
					//	 torsion_size,
					//	 max_bfgs_steps);
				}

				// set
				if (tmp.e < best_e)
				{
					set_update(&tmp, &m_gpu_ligand.rigid, &m_coords,
						m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);

					best_out = tmp;

					// printf("best_out -->> e = %f \n",best_out.e);

					// printf("best_out -->> position = %f , %f, %f \n", best_out.position[0], best_out.position[1], best_out.position[2]);

					// printf("best_out -->> orientation = %f , %f, %f,%f \n", best_out.orientation[0], best_out.orientation[1], best_out.orientation[2],best_out.orientation[3]);
					get_heavy_atom_movable_coords_update(&best_out, m_gpu, &best_coords,&m_coords); // get coords
					best_e = tmp.e;
				}
			}
		}
	}
	// write the best conformation back to CPU
	results[id] = best_out;
	coords[id] = best_coords;
}

extern "C" 
void kernel_monte(
	const output_type_cl *ric,	   // delete the __global
	m_cl *mg,					   // delete the __global
	pre_cl *pre,				   // delete the __constant
	grids_cl *grids,			   // delete the __constant
	random_maps *random_maps,	   // delete the __constant
	ligand_atom_coords_cl *coords, // delete the __global
	output_type_cl *results,	   // delete the __global
	const mis_cl *mis,			   // delete the __global
	const int torsion_size,
	const int search_depth,
	const int max_bfgs_steps,
	const int rilc_bfgs_enable)
{
	dim3 grid_size(128, 64, 1);
	dim3 block_size(16, 2, 1);
	cudaEvent_t time1, time2;
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);

	cudaEventRecord(time1, 0);

	kernel2_update<<<grid_size, block_size>>>(ric, mg, pre, grids, random_maps, coords, results, mis, torsion_size,
											  search_depth, max_bfgs_steps, rilc_bfgs_enable);
	checkCUDA(cudaDeviceSynchronize());

	flo kernalExecutionTime;
	cudaError_t error;
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("kernel2 was wrong [%s]\n", cudaGetErrorString(error));
		// if(system("pause")) printf("kernel2 was wrong");
	}
	cudaEventRecord(time2, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);

	cudaEventElapsedTime(&kernalExecutionTime, time1, time2);
	// printf("\nLCF-debug-the-program");
	// printf("\nElapsed time for kernel_monte-GPU calculation: %0.6f s \n", kernalExecutionTime / 1000);

	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
}