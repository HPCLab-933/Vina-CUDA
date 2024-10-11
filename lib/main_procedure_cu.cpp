#include "cache.h"
// #include "wrapcl.h"
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "parallel_mc.h"
#include "szv_grid.h"
#include <thread>
#include <boost/progress.hpp>

#include "commonMacros.h"
// #include "wrapcl.h"
#include "kernel1.cuh"
#include "kernel2.cuh"
#include "random.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <stdio.h>
#include "omp.h"

/**
 * #include <boost/timer/timer.hpp>
 * using boost::timer::cpu_timer
 */
#include <boost/timer.hpp>
boost::timer boost_t;

// extern __constant__ mis_cl mis_cuda;

using namespace std;
volatile enum { FINISH,
				DOCKING,
				ABORT } status;
void print_process()
{
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		sleep(1);
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)
			printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)
			printf(" ");
		printf("|");
		fflush(stdout);

		count++;
		count %= 30;
	} while (status == DOCKING);
	if (status == FINISH)
	{
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("done");
		for (int i = 0; i < 17; i++)
			printf("=");
		printf("|\n");
		fflush(stdout);
	}
	else if (status == ABORT)
	{
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("error");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("|\n");
		fflush(stdout);
	}
}

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[],
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size)
{
	std::vector<output_type> results_vina;
	int num_atoms;

#pragma omp parallel for
	for (int i = 0; i < thread; i++)
	{
		output_type_cl tmp = result_ptr[i];
		ligand_atom_coords_cl tmp_coords = result_coords_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)
			tmp_c.ligands[0].rigid.position[j] = tmp.position[j];
		// Orientation
		qt q(tmp.orientation[0], tmp.orientation[1], tmp.orientation[2], tmp.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp.e);
		// torsion
		for (int j = 0; j < lig_torsion_size; j++)
			tmp_vina.c.ligands[0].torsions.push_back(tmp.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++)
		{
			vec v_tmp(tmp_coords.coords[j][0], tmp_coords.coords[j][1], tmp_coords.coords[j][2]);
			if ((v_tmp[0] != 0 || v_tmp[1] != 0) || (v_tmp[2] != 0))
				tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
		if (i == 0)
			num_atoms = tmp_vina.coords.size();
		if (num_atoms != tmp_vina.coords.size())
		{
			throw std::runtime_error("atom coords not match!");
		}
	}
	return results_vina;
}
// lcf-debug
/*
void main_procedure_cl(cache& c, const std::vector<model>& ms,  const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, std::vector<output_container>& outs, std::string opencl_binary_path,
	const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs) */

void main_procedure_cl(cache &c, const std::vector<model> &ms, const precalculate &p, const parallel_mc par,
					   const vec &corner1, const vec &corner2, const int seed, std::vector<output_container> &outs,
					   const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs)
{

	/**************************************************************************/
	/***************************    GPU Init    ****************************/
	/**************************************************************************/

	// static std::ofstream g_log("out.log");
	// std::cout.rdbuf(g_log.rdbuf());

	int device = 0; // Assuming device 0
	cudaDeviceProp props;
	checkCUDA(cudaGetDeviceProperties(&props, device));
	int max_wg_size = props.maxThreadsPerBlock;

	// std::cout << "Device name: " << props.name << std::endl;
	// std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
	// std::cout << "The maxThreadsPerBlock is : " << max_wg_size << std::endl;

	// int max_wi_size[3];
	// max_wi_size[0] = props.maxThreadsDim[0];
	// max_wi_size[1] = props.maxThreadsDim[1];
	// max_wi_size[2] = props.maxThreadsDim[2];

	// std::cout<<"The maxThreadsPerBlock is : "<<max_wg_size<<std::endl;

	/**************************************************************************/
	/************************    Original Vina code    ************************/
	/**************************************************************************/

	// printf("\nUsing random seed: %d , Search depth is set to:%f \n", seed, par.mc.search_depth);

	sz nat = num_atom_types(c.atu);

	szv needed;
	for (int i = 0; i < nat; i++)
	{
		if (!c.grids[i].initialized())
		{
			needed.push_back(i);
			c.grids[i].init(c.gd);
		}
	}

	flv affinities(needed.size());

	grid &g = c.grids[needed.front()];

	const fl cutoff_sqr = p.cutoff_sqr();

	grid_dims gd_reduced = szv_grid_dims(c.gd);
	szv_grid ig(ms[0], gd_reduced, cutoff_sqr); // use ms[0]
	// szv_grid ig2(ms[1], gd_reduced, cutoff_sqr); // use ms[0]

	for (int i = 0; i < 3; i++)
	{
		if (ig.m_init[i] != g.m_init[i])
		{
			printf("m_init not equal!");
			exit(-1);
		}
		if (ig.m_range[i] != g.m_range[i])
		{
			printf("m_range not equal!");
			exit(-1);
		}
	}

	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl

	std::vector<conf_size> s;
	std::vector<output_type> tmps;
	for (int i = 0; i < ms.size(); i++)
	{
		s.push_back(ms[i].get_size());
		output_type tmp(s[i], 0);
		tmps.push_back(tmp);
	}
	// change g(s);

	// quasi_newton quasi_newton_par; const int quasi_newton_par_max_steps = par.mc.ssd_par.evals;
	/**********************************************************************************/
	/************************ Kernel1  & Allocate GPU memory **************************/
	/**********************************************************************************/
	boost_t.restart();

	checkCUDA(cudaSetDevice(0));
	cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8;
	checkCUDA(cudaStreamCreate(&stream1));
	checkCUDA(cudaStreamCreate(&stream2));
	checkCUDA(cudaStreamCreate(&stream3));
	checkCUDA(cudaStreamCreate(&stream4));
	checkCUDA(cudaStreamCreate(&stream5));
	checkCUDA(cudaStreamCreate(&stream6));
	checkCUDA(cudaStreamCreate(&stream7));
	checkCUDA(cudaStreamCreate(&stream8));

	rng generator(static_cast<rng::result_type>(seed));
	model m = ms[0];

	// Preparing grid related data
	if (GRIDS_SIZE != c.grids.size())
	{
		throw std::runtime_error("grid_size has to be 17!");
	} // grid_size has to be 17
	grids_cl *grids_ptr = (grids_cl *)malloc(sizeof(grids_cl));
	grid *tmp_grid_ptr = &c.grids[0];

	grids_ptr->atu = c.atu;		// atu
	grids_ptr->slope = c.slope; // slope
	int grids_front;
#pragma omp parallel for
	for (int i = GRIDS_SIZE - 1; i >= 0; i--)
	{
		for (int j = 0; j < 3; j++)
		{
			grids_ptr->grids[i].m_init[j] = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j] = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j] = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0)
		{
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0();
			assert(MAX_NUM_OF_GRID_MI >= grids_ptr->grids[i].m_i);
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1();
			assert(MAX_NUM_OF_GRID_MJ >= grids_ptr->grids[i].m_j);
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2();
			assert(MAX_NUM_OF_GRID_MK >= grids_ptr->grids[i].m_k);
			grids_front = i;
		}
		else
		{
			grids_ptr->grids[i].m_i = 0;
			grids_ptr->grids[i].m_j = 0;
			grids_ptr->grids[i].m_k = 0;
		}
	}
	size_t grids_size = sizeof(grids_cl);

	grids_cl *grids_gpu;
	checkCUDA(cudaMalloc((void **)&grids_gpu, grids_size));
	checkCUDA(cudaMemcpyAsync(grids_gpu, grids_ptr, grids_size, cudaMemcpyHostToDevice, stream1));

	// Preparing atom relationship
	if (MAX_NUM_OF_ATOM_RELATION_COUNT < ig.m_data.m_data.size())
	{
		throw std::runtime_error("relation too large! please use -DLARGE_BOX in makefile");
	}
	assert(ig.m_data.m_i <= 10);
	assert(ig.m_data.m_j <= 10);
	assert(ig.m_data.m_k <= 10);
	ar_cl *ar_ptr = (ar_cl *)malloc(sizeof(ar_cl));
	for (int i = 0; i < ig.m_data.m_data.size(); i++)
	{
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		if (MAX_NUM_OF_ATOM_RELATION_COUNT < ar_ptr->relation_size[i])
		{
			throw std::runtime_error("relation too large!");
		}
		for (int j = 0; j < ar_ptr->relation_size[i]; j++)
		{
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
		}
	}
	size_t ar_size = sizeof(ar_cl);

	ar_cl *ar_gpu;
	checkCUDA(cudaMalloc((void **)&ar_gpu, ar_size));
	checkCUDA(cudaMemcpyAsync(ar_gpu, ar_ptr, ar_size, cudaMemcpyHostToDevice, stream2));

	// Preparing protein atoms related data
	pa_cl *pa_ptr = (pa_cl *)malloc(sizeof(pa_cl));
	if (MAX_NUM_OF_PROTEIN_ATOMS <= m.grid_atoms.size())
	{
		throw std::runtime_error("pocket too large!");
	}
	for (int i = 0; i < m.grid_atoms.size(); i++)
	{
		pa_ptr->atoms[i].types[0] = m.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = m.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = m.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = m.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++)
			pa_ptr->atoms[i].coords[j] = m.grid_atoms[i].coords.data[j];
	}
	size_t pa_size = sizeof(pa_cl);

	pa_cl *pa_gpu;
	checkCUDA(cudaMalloc((void **)&pa_gpu, pa_size));
	checkCUDA(cudaMemcpyAsync(pa_gpu, pa_ptr, pa_size, cudaMemcpyHostToDevice, stream3));

	// Preaparing precalculated look up table related data
	pre_cl *pre_ptr = (pre_cl *)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if (MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size())
	{
		throw std::runtime_error("LUT too large!");
	}

	for (int i = 0; i < p.data.m_data.size(); i++)
	{
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		if (FAST_SIZE != p.data.m_data[i].fast.size())
		{
			throw std::runtime_error("fast too large!");
		}
		if (SMOOTH_SIZE != p.data.m_data[i].smooth.size())
		{
			throw std::runtime_error("smooth too large!");
		}

		for (int j = 0; j < FAST_SIZE; j++)
		{
			pre_ptr->m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++)
		{
			pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	size_t pre_size = sizeof(pre_cl);

	pre_cl *pre_gpu;
	checkCUDA(cudaMalloc((void **)&pre_gpu, pre_size));
	checkCUDA(cudaMemcpyAsync(pre_gpu, pre_ptr, pre_size, cudaMemcpyHostToDevice, stream4));

	// Preparing grid boundries
	gb_cl *gb_ptr = (gb_cl *)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0();
	gb_ptr->dims[1] = ig.m_data.dim1();
	gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++)
		gb_ptr->init[i] = ig.m_init.data[i];
	for (int i = 0; i < 3; i++)
		gb_ptr->range[i] = ig.m_range.data[i];
	size_t gb_size = sizeof(gb_cl);

	gb_cl *gb_gpu;
	checkCUDA(cudaMalloc((void **)&gb_gpu, gb_size));
	checkCUDA(cudaMemcpyAsync(gb_gpu, gb_ptr, gb_size, cudaMemcpyHostToDevice, stream5));

	// miscellaneous
	mis_cl *mis_ptr = (mis_cl *)malloc(sizeof(mis_cl));
	mis_ptr->needed_size = needed.size();
	mis_ptr->epsilon_fl = epsilon_fl;
	mis_ptr->cutoff_sqr = cutoff_sqr;
	mis_ptr->max_fl = max_fl;
	// mis_ptr->torsion_size = tmp.c.ligands[0].torsions.size();
	// mis_ptr->max_bfgs_steps = quasi_newton_par_max_steps;
	// mis_ptr->search_depth = par.mc.search_depth;
	mis_ptr->mutation_amplitude = par.mc.mutation_amplitude;
	// mis_ptr->total_wi = max_wi_size[0] * max_wi_size[1];
	mis_ptr->thread = par.mc.thread;
	mis_ptr->ar_mi = ig.m_data.m_i;
	mis_ptr->ar_mj = ig.m_data.m_j;
	mis_ptr->ar_mk = ig.m_data.m_k;
	mis_ptr->grids_front = grids_front;
	for (int i = 0; i < 3; i++)
		mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++)
		mis_ptr->hunt_cap[i] = par.mc.hunt_cap[i];
	size_t mis_size = sizeof(mis_cl);

	mis_cl *mis_gpu;
	checkCUDA(cudaMalloc((void **)&mis_gpu, mis_size));
	checkCUDA(cudaMemcpyAsync(mis_gpu, mis_ptr, mis_size, cudaMemcpyHostToDevice, stream6));

	float *needed_ptr = (float *)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < mis_ptr->needed_size; i++)
		needed_ptr[i] = needed[i];

	float *needed_gpu;
	checkCUDA(cudaMalloc((void **)&needed_gpu, mis_ptr->needed_size * sizeof(float)));
	checkCUDA(cudaMemcpyAsync(needed_gpu, needed_ptr, mis_ptr->needed_size * sizeof(float), cudaMemcpyHostToDevice, stream7));

	/**************************************************************************/
	/************************    Allocate GPU memory    ***********************/
	/**************************************************************************/
	// cudaError_t err;
	// cudaStream_t stream = 0;
	// checkCUDA(cudaStreamCreate(&stream));

	/**
	 *  extern boost::timer::cpu_timer boost
	 */
	/*
	pre_cl *pre_gpu;
	checkCUDA(cudaMalloc((void **)&pre_gpu, pre_size));
	// cudaMemcpy(pre_gpu, pre_ptr, pre_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(pre_gpu, pre_ptr, pre_size, cudaMemcpyHostToDevice));

	pa_cl *pa_gpu;
	checkCUDA(cudaMalloc((void **)&pa_gpu, pa_size));
	// cudaMemcpy(pa_gpu, pa_ptr, pa_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(pa_gpu, pa_ptr, pa_size, cudaMemcpyHostToDevice));

	gb_cl *gb_gpu;
	checkCUDA(cudaMalloc((void **)&gb_gpu, gb_size));
	// cudaMemcpy(gb_gpu, gb_ptr, gb_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(gb_gpu, gb_ptr, gb_size, cudaMemcpyHostToDevice));

	ar_cl *ar_gpu;
	checkCUDA(cudaMalloc((void **)&ar_gpu, ar_size));
	// cudaMemcpy(ar_gpu, ar_ptr, ar_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(ar_gpu, ar_ptr, ar_size, cudaMemcpyHostToDevice));

	grids_cl *grids_gpu;
	checkCUDA(cudaMalloc((void **)&grids_gpu, grids_size));
	// cudaMemcpy(grids_gpu, grids_ptr, grids_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(grids_gpu, grids_ptr, grids_size, cudaMemcpyHostToDevice));

	float *needed_gpu;
	checkCUDA(cudaMalloc((void **)&needed_gpu, mis_ptr->needed_size * sizeof(float)));
	// cudaMemcpy(needed_gpu, needed_ptr, mis_ptr->needed_size * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(needed_gpu, needed_ptr, mis_ptr->needed_size * sizeof(float), cudaMemcpyHostToDevice));

	mis_cl *mis_gpu;
	checkCUDA(cudaMalloc((void **)&mis_gpu, mis_size));
	// cudaMemcpy(mis_gpu, mis_ptr, mis_size, cudaMemcpyHostToDevice);
	checkCUDA(cudaMemcpy(mis_gpu, mis_ptr, mis_size, cudaMemcpyHostToDevice));
	*/

	// Make sure all streams have finished their work before continuing
	// checkCUDA(cudaStreamSynchronize(stream1));
	// checkCUDA(cudaStreamSynchronize(stream2));
	// checkCUDA(cudaStreamSynchronize(stream3));
	// checkCUDA(cudaStreamSynchronize(stream4));
	checkCUDA(cudaStreamSynchronize(stream5));
	// checkCUDA(cudaStreamSynchronize(stream6));
	// checkCUDA(cudaStreamSynchronize(stream7));

	// Destroy streams when they are no longer needed
	checkCUDA(cudaStreamDestroy(stream1));
	checkCUDA(cudaStreamDestroy(stream2));
	checkCUDA(cudaStreamDestroy(stream3));
	checkCUDA(cudaStreamDestroy(stream4));
	checkCUDA(cudaStreamDestroy(stream5));
	checkCUDA(cudaStreamDestroy(stream6));
	checkCUDA(cudaStreamDestroy(stream7));

	// std::cout << "time of memcpy for kernel_grid: " << boost_t.elapsed() << std::endl;
	status = DOCKING;
#ifdef NDEBUG
	std::thread console_thread(print_process);
#endif

	/**************************************************************************/
	/************************  Call to kernel function  ***********************/
	/**************************************************************************/

	// std::cout << "<<Call to kernel_grid function>> " << std::endl;
	kernel_grid(pre_gpu, pa_gpu, gb_gpu, ar_gpu, grids_gpu, mis_gpu, needed_gpu, c.atu, nat);

	free(pa_ptr);
	free(gb_ptr);
	free(ar_ptr);
	free(needed_ptr);
	free(pre_ptr);
	free(grids_ptr);

	checkCUDA(cudaFree(pa_gpu));
	checkCUDA(cudaFree(gb_gpu));
	checkCUDA(cudaFree(ar_gpu));
	checkCUDA(cudaFree(needed_gpu));

	/**************************************************************************/
	/************************    Kernel2    ***********************/
	/**************************************************************************/
	int num_ligands = ms.size();
	std::vector<random_maps *> rand_maps_ptrs(num_ligands);
	std::vector<output_type_cl *> ric_ptrs(num_ligands);
	std::vector<m_cl *> m_ptrs(num_ligands);
	std::vector<ligand_atom_coords_cl *> result_coords_ptrs(num_ligands);
	std::vector<output_type_cl *> result_ptrs(num_ligands);
	std::vector<int> torsion_sizes(num_ligands);

	// std::vector<cl_mem> ric_gpus			(num_ligands);
	// std::vector<cl_mem> m_gpus				(num_ligands);
	// std::vector<cl_mem> random_maps_gpus	(num_ligands);
	// std::vector<cl_mem> result_coords_gpus	(num_ligands);
	// std::vector<cl_mem> result_gpus			(num_ligands);

	std::vector<output_type_cl *> ric_gpus(num_ligands);
	std::vector<m_cl *> m_gpus(num_ligands);
	std::vector<random_maps *> random_maps_gpus(num_ligands);
	std::vector<ligand_atom_coords_cl *> result_coords_gpus(num_ligands);
	std::vector<output_type_cl *> result_gpus(num_ligands);

	// std::vector<cl_event> ligands_events	(num_ligands);

	for (int ligand_count = 0; ligand_count < num_ligands; ligand_count++)
	{

		// printf("\nDocking ligand: name: [%s] , the Atom size is: [%lu] , the torsion_sizes is : [%lu] \n", ligand_names[ligand_count][0].c_str(), ms[ligand_count].atoms.size(), tmps[ligand_count].c.ligands[0].torsions.size());
		try
		{
			model m = ms[ligand_count];
			if (m.atoms.size() >= MAX_NUM_OF_ATOMS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_ATOMS too large!");
			}

			// assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

			output_type tmp = tmps[ligand_count];
			torsion_sizes[ligand_count] = tmp.c.ligands[0].torsions.size();
			if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_LIG_TORSION too large!");
			}

			// assert(tmp.c.ligands[0].torsions.size() < MAX_NUM_OF_LIG_TORSION);

			// Preparing m related data
			m_ptrs[ligand_count] = (m_cl *)malloc(sizeof(m_cl));
			m_cl *m_ptr = m_ptrs[ligand_count];

			for (int i = 0; i < m.atoms.size(); i++)
			{
				m_ptr->atoms[i].types[0] = m.atoms[i].el; // To store 4 atoms types (el, ad, xs, sy)
				m_ptr->atoms[i].types[1] = m.atoms[i].ad;
				m_ptr->atoms[i].types[2] = m.atoms[i].xs;
				m_ptr->atoms[i].types[3] = m.atoms[i].sy;
				for (int j = 0; j < 3; j++)
				{
					m_ptr->atoms[i].coords[j] = m.atoms[i].coords[j]; // To store atom coords
				}
			}

			// To store atoms coords
			for (int i = 0; i < m.coords.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					m_ptr->m_coords.coords[i][j] = m.coords[i].data[j];
				}
			}

			// To store minus forces
			for (int i = 0; i < m.coords.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					m_ptr->minus_forces.coords[i][j] = m.minus_forces[i].data[j];
				}
			}

			// Generate random maps
			rand_maps_ptrs[ligand_count] = (random_maps *)malloc(sizeof(random_maps));
			random_maps *rand_maps_ptr = rand_maps_ptrs[ligand_count];

#pragma omp parallel for
			for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++)
			{
				rand_maps_ptr->int_map[i] = random_int(0, int(tmp.c.ligands[0].torsions.size()), generator);
				rand_maps_ptr->pi_map[i] = random_fl(-pi, pi, generator);
			}

#pragma omp parallel for
			for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++)
			{
				vec rand_coords = random_inside_sphere(generator);
				for (int j = 0; j < 3; j++)
				{
					rand_maps_ptr->sphere_map[i][j] = rand_coords[j];
				}
			}
			size_t rand_maps_size = sizeof(*rand_maps_ptr);

			// Preparing random initial conformations
			ric_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
			output_type_cl *ric_ptr = ric_ptrs[ligand_count];

#pragma omp parallel for
			for (int i = 0; i < par.mc.thread; i++)
			{
				tmp.c.randomize(corner1, corner2, generator);
				for (int j = 0; j < 3; j++)
				{
					ric_ptr[i].position[j] = tmp.c.ligands[0].rigid.position[j];
				}
				ric_ptr[i].orientation[0] = tmp.c.ligands[0].rigid.orientation.R_component_1();
				ric_ptr[i].orientation[1] = tmp.c.ligands[0].rigid.orientation.R_component_2();
				ric_ptr[i].orientation[2] = tmp.c.ligands[0].rigid.orientation.R_component_3();
				ric_ptr[i].orientation[3] = tmp.c.ligands[0].rigid.orientation.R_component_4();
				if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION)
				{
					throw std::runtime_error("/lib/main_procedure_cu.cpp: NUM_OF_LIG_TORSION is too large");
				}
				for (int j = 0; j < tmp.c.ligands[0].torsions.size(); j++)
					ric_ptr[i].lig_torsion[j] = tmp.c.ligands[0].torsions[j];

				if ((tmp.c.flex.size() != 0))
				{
					assert(tmp.c.flex[0].torsions.size() < MAX_NUM_OF_FLEX_TORSION);
					for (int j = 0; j < tmp.c.flex[0].torsions.size(); j++)
						ric_ptr[i].flex_torsion[j] = tmp.c.flex[0].torsions[j];
				}
			}

			size_t ric_size = par.mc.thread * sizeof(output_type_cl);

			// Preparing ligand data
			if (m.num_other_pairs() != 0)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: m.other_paris is not supported!");
			} // m.other_paris is not supported!
			if (m.ligands.size() != 1)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Only one ligand supported!");
			} // Only one ligand supported!
			m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
			// std::cout << "m_ptr->ligand.pairs.num_pairs = " << m_ptr->ligand.pairs.num_pairs << std::endl;
			if (m.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_LIG_PAIRS Too Large!");
			}
			// std::cout << "m_ptr->ligand.pairs.num_pairs = " << m_ptr->ligand.pairs.num_pairs << std::endl;
#pragma omp parallel for
			for (int i = 0; i < m_ptr->ligand.pairs.num_pairs; i++)
			{
				m_ptr->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
				m_ptr->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
				m_ptr->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
			}

			m_ptr->ligand.begin = m.ligands[0].begin; // 0
			m_ptr->ligand.end = m.ligands[0].end;	  // 29
			ligand m_ligand = m.ligands[0];			  // Only support one ligand
			if (m_ligand.end >= MAX_NUM_OF_ATOMS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: NUM_OF_ATOMS Ligand Too large!");
			}

			// Store root node
			m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
			m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
			for (int i = 0; i < 3; i++)
				m_ptr->ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];

#pragma omp parallel for
			for (int i = 0; i < 9; i++)
			{
				m_ptr->ligand.rigid.orientation_m[0][i] = m_ligand.node.get_orientation_m().data[i];
			}

			m_ptr->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
			m_ptr->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
			m_ptr->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
			m_ptr->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();

			for (int i = 0; i < 3; i++)
			{
				m_ptr->ligand.rigid.axis[0][i] = 0;
				m_ptr->ligand.rigid.relative_axis[0][i] = 0;
				m_ptr->ligand.rigid.relative_origin[0][i] = 0;
			}

			// Store children nodes (in depth-first order)
			struct tmp_struct
			{
				int start_index = 0;
				int parent_index = 0;
				void store_node(tree<segment> &child_ptr, rigid_cl &rigid)
				{
					start_index++; // start with index 1, index 0 is root node
					rigid.parent[start_index] = parent_index;
					rigid.atom_range[start_index][0] = child_ptr.node.begin;
					rigid.atom_range[start_index][1] = child_ptr.node.end;
					for (int i = 0; i < 9; i++)
					{
						rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
					}
					rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
					rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
					rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
					rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
					for (int i = 0; i < 3; i++)
					{
						rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
						rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
						rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
						rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
					}
					if (child_ptr.children.size() == 0)
						return;
					else
					{
						if (start_index >= MAX_NUM_OF_RIGID)
						{
							throw std::runtime_error("/lib/main_procedure_cu.cpp: Children map too large!");
						}
						int parent_index_tmp = start_index;
						for (int i = 0; i < child_ptr.children.size(); i++)
						{
							this->parent_index = parent_index_tmp; // Update parent index
							this->store_node(child_ptr.children[i], rigid);
						}
					}
				}
			};
			tmp_struct ts;
			for (int i = 0; i < m_ligand.children.size(); i++)
			{
				ts.parent_index = 0; // Start a new branch, whose parent is 0
				ts.store_node(m_ligand.children[i], m_ptr->ligand.rigid);
			}
			m_ptr->ligand.rigid.num_children = ts.start_index;
			// std::cout << "m_ptr->ligand.rigid.num_children [num_torsion] = " << ts.start_index << std::endl;

			// set children_map
#pragma omp parallel for
			for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
			{
				for (int j = 0; j < MAX_NUM_OF_RIGID; j++)
				{
					m_ptr->ligand.rigid.children_map[i][j] = false;
				}
			}

#pragma omp parallel for
			for (int i = 1; i < m_ptr->ligand.rigid.num_children + 1; i++)
			{
				int parent_index = m_ptr->ligand.rigid.parent[i];
				m_ptr->ligand.rigid.children_map[parent_index][i] = true;
			}
			m_ptr->m_num_movable_atoms = m.num_movable_atoms();
			// std::cout << "m_ptr->m_num_movable_atoms = " << m_ptr->m_num_movable_atoms << std::endl;
			size_t m_size = sizeof(m_cl);

			// printf("LCF-DEBUG the m_ptr-773: %d \n",m_ptr->atoms[1].types[3]);
			// printf("LCF-DEBUG the two pre->m_cutoff_sqrr: %f \n",pre_ptr->m_cutoff_sqr);
			// std::cout<<"LCF-DEBUG the m_ptr-773: \n"<<m_ptr->atoms[1].types[3]<<std::endl;
			// std::cout<<"LCF-DEBUG the two pre->m_cutoff_sqrr:  \n"<<pre_ptr->m_cutoff_sqr<<std::endl;
			// std::cout<<"completed the computing the grid ====> then for mento_computing"<<std::endl;

			// Init ligand atom coords
			result_coords_ptrs[ligand_count] = (ligand_atom_coords_cl *)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));
			// memset(result_coords_ptrs[ligand_count], 0, sizeof(ligand_atom_coords_cl));

			// Init results
			result_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
			// memset(result_ptrs[ligand_count],0,sizeof(output_type_cl));
			// cl_mem ric_gpu;
			// output_type_cl* ric_gpus[ligand_count];
			boost_t.restart();
			checkCUDA(cudaMalloc((void **)&ric_gpus[ligand_count], ric_size));
			// checkCUDA(cudaMemcpy(ric_gpus[ligand_count], ric_ptr, ric_size, cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(ric_gpus[ligand_count], ric_ptr, ric_size, cudaMemcpyHostToDevice));

			// checkCUDA(cudaMalloc((void **)&ric_gpus[ligand_count], ric_size));

			// for (int i = 0; i < mis_ptr->thread; i++)
			//{
			//	checkCUDA(cudaMemcpy(ric_gpus[ligand_count] + i, ric_ptr, sizeof(output_type_cl), cudaMemcpyHostToDevice));
			// }

			// CreateDeviceBuffer(&ric_gpus[ligand_count], CL_MEM_READ_ONLY, ric_size, context);
			// err = clEnqueueWriteBuffer(queue, ric_gpus[ligand_count], false, 0, ric_size, ric_ptr, 0, NULL, NULL); checkErr(err);

			// m_cl* m_gpus[ligand_count];
			// checkCUDA(cudaMalloc((void **)&m_gpus[ligand_count], m_size));
			// checkCUDA(cudaMemcpy(m_gpus[ligand_count], m_ptr, m_size, cudaMemcpyHostToDevice));
			// checkCUDA(cudaMemcpyAsync(m_gpus[ligand_count], m_ptr, m_size, cudaMemcpyHostToDevice, stream));
			// std::cout << "the ric_gpus[ligand_count] memory size = " << ric_size << std::endl;
			// std::cout << "the m_gpus[ligand_count] memory size =  " << mis_ptr->thread * m_size << std::endl;

			checkCUDA(cudaMalloc((void **)&m_gpus[ligand_count], mis_ptr->thread * m_size));
#pragma omp parallel for
			for (int i = 0; i < mis_ptr->thread; i++)
			{
				checkCUDA(cudaMemcpy(m_gpus[ligand_count] + i, m_ptr, m_size, cudaMemcpyHostToDevice));
			}

			// cl_mem m_gpu;
			// CreateDeviceBuffer(&m_gpus[ligand_count], CL_MEM_READ_WRITE, m_size, context);
			// err = clEnqueueWriteBuffer(queue, m_gpus[ligand_count], false, 0, m_size, m_ptr, 0, NULL, NULL); checkErr(err);

			// random_maps* random_maps_gpus[ligand_count];
			checkCUDA(cudaMalloc((void **)&random_maps_gpus[ligand_count], rand_maps_size));
			// checkCUDA(cudaMemcpy(random_maps_gpus[ligand_count], rand_maps_ptr, rand_maps_size, cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(random_maps_gpus[ligand_count], rand_maps_ptr, rand_maps_size, cudaMemcpyHostToDevice));

			// cl_mem random_maps_gpu;
			// CreateDeviceBuffer(&random_maps_gpus[ligand_count], CL_MEM_READ_ONLY, rand_maps_size, context);
			// err = clEnqueueWriteBuffer(queue, random_maps_gpus[ligand_count], false, 0, rand_maps_size, rand_maps_ptr, 0, NULL, NULL); checkErr(err);

			// ligand_atom_coords_cl* result_coords_gpus[ligand_count];
			//  cudamemcheck
			checkCUDA(cudaMalloc((void **)&result_coords_gpus[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl)));
			// checkCUDA(cudaMemcpy(result_coords_gpus[ligand_count], result_coords_ptrs[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(result_coords_gpus[ligand_count], result_coords_ptrs[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl), cudaMemcpyHostToDevice));
			// cl_mem result_coords_gpu;
			// CreateDeviceBuffer(&result_coords_gpus[ligand_count], CL_MEM_WRITE_ONLY, mis_ptr->thread * sizeof(ligand_atom_coords_cl), context);
			// err = clEnqueueWriteBuffer(queue, result_coords_gpus[ligand_count], false, 0, mis_ptr->thread * sizeof(ligand_atom_coords_cl),
			//	result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

			// output_type_cl* result_gpus[ligand_count];
			checkCUDA(cudaMalloc((void **)&result_gpus[ligand_count], par.mc.thread * sizeof(output_type_cl)));
			// checkCUDA(cudaMemcpy(result_gpus[ligand_count], result_ptrs[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyHostToDevice));
			checkCUDA(cudaMemcpy(result_gpus[ligand_count], result_ptrs[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyHostToDevice));
			// std::cout << "time of memcpy for kernel monte carlo: " << boost_t.elapsed() << std::endl;

			/**************************************************************************/
			/************************  Call to kernel function  ***********************/
			/**************************************************************************/
			// Call to kernel function
			/*
			printf("e: %f\n", ric_gpus[ligand_count]->e);
			printf("position: (%f, %f, %f)\n", ric_gpus[ligand_count]->position[0], ric_gpus[ligand_count]->position[1], ric_gpus[ligand_count]->position[2]);
			printf("orientation: (%f, %f, %f, %f)\n", ric_gpus[ligand_count]->orientation[0], ric_gpus[ligand_count]->orientation[1], ric_gpus[ligand_count]->orientation[2], ric_gpus[ligand_count]->orientation[3]);

			std::cout << "pre_gpu_res: " << std::endl;
			std::cout << "n: " << pre_gpu_res->n << std::endl;
			std::cout << "m_cutoff_sqr: " << pre_gpu_res->m_cutoff_sqr << std::endl;
			std::cout << "factor: " << pre_gpu_res->factor << std::endl;*/
			// std::cout <<"mis->thread: "<<mis_ptr->thread<< "mis->total_wi: " <<mis_ptr->total_wi<<std::endl;
			// std::cout << "<<Call to kernel_monte function>> " << std::endl;
			// std::cout << "par.mc.search_depth[ligand_count] = " << par.mc.search_depth[ligand_count] << "  || par.mc.ssd_par.bfgs_steps[ligand_count] = " << par.mc.ssd_par.bfgs_steps[ligand_count] << std::endl;
			// std::cout << "ric_size: " << sizeof(*ric_gpus[ligand_count]) * par.mc.thread << "\n"
			//		  << "m_size: " << sizeof(*m_gpus[ligand_count]) << "\n"
			//		  << "pre_size: " << sizeof(*pre_gpu) << "\n"
			//		  << "grids_size: " << sizeof(*grids_gpu) << "\n"
			//		  << "random_maps_size: " << sizeof(*random_maps_gpus[ligand_count]) << "\n"
			//		  << "result_coords_size: " << sizeof(*result_coords_gpus[ligand_count]) * par.mc.thread << "\n"
			//		  << "result_size: " << sizeof(*result_gpus[ligand_count]) * par.mc.thread << "\n"
			//		  << "mis_size: " << sizeof(*mis_gpu) << "\n"
			//		  << "torsion_size_size: " << sizeof(torsion_sizes[ligand_count]) << "\n"
			//		  << "par_mc_search_depth_size: " << sizeof(par.mc.search_depth[ligand_count]) << "\n"
			//		  << "par_mc_ssd_par_bfgs_steps_size: " << sizeof(par.mc.ssd_par.bfgs_steps[ligand_count]) << "\n"
			//		  << "rilc_bfgs_size: " << sizeof(rilc_bfgs) << std::endl;

			kernel_monte(ric_gpus[ligand_count], m_gpus[ligand_count], pre_gpu, grids_gpu, random_maps_gpus[ligand_count],
						 result_coords_gpus[ligand_count], result_gpus[ligand_count], mis_gpu, torsion_sizes[ligand_count],
						 par.mc.search_depth[ligand_count], par.mc.ssd_par.bfgs_steps[ligand_count], rilc_bfgs);

			// err = clEnqueueNDRangeKernel(queue, kernels[1], 2, 0, kernel2_global_size, kernel2_local_size,
			//	0, NULL, &ligands_events[ligand_count]); checkErr(err);

			// clWaitForEvents(1, &ligands_events[ligand_count]);

			// Init ligand atom coords
			result_coords_ptrs[ligand_count] = (ligand_atom_coords_cl *)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));

			// Init results
			result_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
			// std::vector<output_type_cl*>		result_ptrs			(num_ligands);
			// std::vector<ligand_atom_coords_cl*> result_coords_ptrs	(num_ligands);

			checkCUDA(cudaMemcpy(result_ptrs[ligand_count], result_gpus[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost));

			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));

			// err = clEnqueueReadBuffer(queue, result_gpus[ligand_count], false, 0, par.mc.thread * sizeof(output_type_cl),
			//	result_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

			// err = clEnqueueReadBuffer(queue, result_coords_gpus[ligand_count], false, 0, par.mc.thread * sizeof(ligand_atom_coords_cl),
			//	result_coords_ptrs[ligand_count], 0, NULL, NULL); checkErr(err);

			// clFinish(queue);

			std::vector<output_type> result_vina = cl_to_vina(result_ptrs[ligand_count], result_coords_ptrs[ligand_count],
															  par.mc.thread, torsion_sizes[ligand_count]);
			// if empty, something goes wrong in the device part
			if (result_vina.size() == 0)
			{
				status = ABORT;
#ifdef NDEBUG
				console_thread.join();
#endif
				exit(-1);
			}

			// Write back to vina
#pragma omp parallel for
			for (int i = 0; i < par.mc.thread; i++)
			{
				// assert(result_vina[i].coords.size()== mis_ptr->)
				add_to_output_container(outs[ligand_count], result_vina[i], par.mc.min_rmsd, par.mc.num_saved_mins);
			}
			// VINA_CHECK(!outs[ligand_count].empty());
			// VINA_CHECK(outs[ligand_count].front().e <= outs[ligand_count].back().e);

			free(rand_maps_ptrs[ligand_count]);
			free(ric_ptrs[ligand_count]);
			free(m_ptrs[ligand_count]);
			free(result_coords_ptrs[ligand_count]);
			free(result_ptrs[ligand_count]);

			checkCUDA(cudaFree(result_coords_gpus[ligand_count]));
			checkCUDA(cudaFree(result_gpus[ligand_count]));
			checkCUDA(cudaFree(ric_gpus[ligand_count]));
			checkCUDA(cudaFree(m_gpus[ligand_count]));
			checkCUDA(cudaFree(random_maps_gpus[ligand_count]));

			// err = clReleaseMemObject(result_coords_gpus[ligand_count]); checkErr(err);
			// err = clReleaseMemObject(result_gpus[ligand_count]);		checkErr(err);
			// err = clReleaseMemObject(ric_gpus[ligand_count]);			checkErr(err);
			// err = clReleaseMemObject(m_gpus[ligand_count]);				checkErr(err);
			// err = clReleaseMemObject(random_maps_gpus[ligand_count]);	checkErr(err);

			// #ifndef TIME_ANALYSIS
			//		err = clReleaseEvent(ligands_events[ligand_count]);			checkErr(err);
			// #endif // !TIME_ANALYSIS
		}
		catch (...)
		{
			continue;
		}
	}

	free(mis_ptr);
	checkCUDA(cudaFree(mis_gpu));
	checkCUDA(cudaFree(grids_gpu));
	checkCUDA(cudaFree(pre_gpu));
	// free(mis_gpu_res);
	// free(grids_gpu_res);
	// free(pre_gpu_res);

	// err = clReleaseMemObject(mis_gpu);		checkErr(err);
	// err = clReleaseMemObject(grids_gpu);	checkErr(err);
	// err = clReleaseMemObject(pre_gpu);		checkErr(err);

	status = FINISH;
#ifdef NDEBUG
	console_thread.join(); // wait the thread finish
#endif
#ifdef TIME_ANALYSIS
	// Output Analysis
	cl_ulong time_start, time_end;
	double total_time = 0;
	std::string log_file = "gpu_runtime.log";
	// delete file if file exists
	if (std::ifstream(log_file))
	{
		std::remove(log_file.c_str());
	}
	for (int ligand_count = 0; ligand_count < ms.size(); ligand_count++)
	{
		cl_event event_tmp = ligands_events[ligand_count];
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		checkErr(err);
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		checkErr(err);
		total_time = time_end - time_start;
		err = clReleaseEvent(ligands_events[ligand_count]);
		checkErr(err);
		printf("\nAutoDockVina-GPU3 ligand %d runtime = %0.3f s", ligand_count, (total_time / 1000000000.0));

		std::cout << std::setiosflags(std::ios::fixed);
		std::ofstream file(log_file, std::ios::app);
		if (file.is_open())
		{
			file << "AutoDockVina-GPU3 monte carlo runtime of ligand " << ligand_names[ligand_count][0] << " = " << std::setprecision(5) << (total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
	}
	printf("\n");
#endif
}
/*
#ifdef OPENCL_TIME_ANALYSIS
		cout << setiosflags(ios::fixed);
		std::ofstream file("gpu_runtime.log");
		if (file.is_open())
		{
			file << "GPU grid cache runtime = " << (kernel1_total_time / 1000000000.0) << " s" << std::endl;
			file << "GPU monte carlo runtime = " << (kernel2_total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
#endif
}*/