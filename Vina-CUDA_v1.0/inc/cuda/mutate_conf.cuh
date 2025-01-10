#ifndef _MUTATE_CONF_H_
#define _MUTATE_CONF_H_
#include "commonMacros.h"

__device__ void quaternion_increment(flo* q, const flo* rotation, flo epsilon_fl);
__device__ inline void normalize_angle(flo* x);

__device__ void output_type_cl_increment(output_type_cl* x, const change_cl* c, flo factor, flo epsilon_fl, int lig_torsion_size) {
	
	flo rotation[3];
	// position increment and orientation increment
	for (int k = 0; k < 3; k++) {
		x->position[k] += factor * c->position[k];
		rotation[k] = factor * c->orientation[k];
	}
	quaternion_increment(x->orientation, rotation, epsilon_fl);
	
	// torsion increment
	for (int k = 0; k < lig_torsion_size; k++) {
		flo tmp = factor * c->lig_torsion[k];
		normalize_angle(&tmp);
		x->lig_torsion[k] += tmp;
		normalize_angle(&(x->lig_torsion[k]));
	}
}

__device__ inline flo norm3(const flo* a) {
	return sqrtf(powf(a[0], 2) + powf(a[1], 2) + powf(a[2], 2));
}

__device__ inline void normalize_angle(flo* x) {
	while (1) {
		if (*x >= -(M_PI) && *x <= (M_PI)) {
			break;
		}
		else if (*x > 3 * (M_PI)) {
			flo n = (*x - (M_PI)) / (2 * (M_PI));
			*x -= 2 * (M_PI) * ceil(n);
		}
		else if (*x < 3 * -(M_PI)) {
			flo n = (-*x - (M_PI)) / (2 * (M_PI));
			*x += 2 * (M_PI) * ceil(n);
		}
		else if (*x > (M_PI)) {
			*x -= 2 * (M_PI);
		}
		else if (*x < -(M_PI)) {
			*x += 2 * (M_PI);
		}
		else {
			break;
		}
	}
}

__device__ inline void normalize_angle_update(float* x) {
    *x = fmodf(*x + M_PI, 2 * M_PI) - M_PI;
}


__device__ inline bool quaternion_is_normalized(flo* q) {
	flo q_pow = powf(q[0], 2) + powf(q[1], 2) + powf(q[2], 2) + powf(q[3], 2);
	flo sqrt_q_pow = sqrt(q_pow);
	return (q_pow - 1 < 0.001) && (sqrt_q_pow - 1 < 0.001);
}

__device__ inline void angle_to_quaternion(flo* q, const flo* rotation, flo epsilon_fl) {
	//flo angle = norm3(rotation);
	flo angle = sqrtf(powf(rotation[0], 2) + powf(rotation[1], 2) + powf(rotation[2], 2));
	if (angle > epsilon_fl) {
		flo axis[3] = { rotation[0] / angle, rotation[1] / angle ,rotation[2] / angle };
		if (sqrtf(powf(axis[0], 2) + powf(axis[1], 2) + powf(axis[2], 2)) - 1 >= 0.001) printf("\nmutate: angle_to_quaternion() ERROR!"); // Replace assert(eq(axis.norm(), 1));
		normalize_angle(&angle);
		flo c = cos(angle / 2);
		flo s = sin(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
	
	q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0;
	
	return;
}

__device__ inline void angle_to_quaternion_update(flo* q, const flo* rotation, flo epsilon_fl) {
	flo a0 = rotation[0], a1 = rotation[1], a2 = rotation[2];
	flo angle = sqrtf(powf(a0, 2) + powf(a1, 2) + powf(a2, 2));
	if (angle > epsilon_fl) {
		flo axis[3] = { a0 / angle, a1 / angle , a2 / angle };
		if (norm3(axis) - 1 >= 0.001) printf("\nmutate: angle_to_quaternion() ERROR!"); // Replace assert(eq(axis.norm(), 1));
		normalize_angle(&angle);
		flo c = cos(angle / 2);
		flo s = sin(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
}

// quaternion multiplication
__device__ inline void angle_to_quaternion_multi(flo* qa, const flo* qb) {
	flo tmp[4] = { qa[0],qa[1],qa[2],qa[3] };
	qa[0] = tmp[0] * qb[0] - tmp[1] * qb[1] - tmp[2] * qb[2] - tmp[3] * qb[3];
	qa[1] = tmp[0] * qb[1] + tmp[1] * qb[0] + tmp[2] * qb[3] - tmp[3] * qb[2];
	qa[2] = tmp[0] * qb[2] - tmp[1] * qb[3] + tmp[2] * qb[0] + tmp[3] * qb[1];
	qa[3] = tmp[0] * qb[3] + tmp[1] * qb[2] - tmp[2] * qb[1] + tmp[3] * qb[0];
}


__device__ inline void angle_to_quaternion_multi_update(flo* qa, const flo* qb) {
    flo a0 = qa[0], a1 = qa[1], a2 = qa[2], a3 = qa[3];

    flo b0 = qb[0], b1 = qb[1], b2 = qb[2], b3 = qb[3];

    qa[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
    qa[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
    qa[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
    qa[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
}

__device__ inline void quaternion_normalize_approx(flo* q, flo epsilon_fl) {
	const flo s = powf(q[0], 2) + powf(q[1], 2) + powf(q[2], 2) + powf(q[3], 2);
	// Omit one assert()
	if (fabs(s - 1) >= TOLERANCE){
		const flo a = sqrt(s);
		if (a <= epsilon_fl) printf("\nmutate: quaternion_normalize_approx ERROR!"); // Replace assert(a > epsilon_fl);
		for (int i = 0; i < 4; i += 2) {
			q[i] *= (1 / a);
			q[i+1] *= (1 / a);
		}
		if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_normalize_approx() ERROR!");// Replace assert(quaternion_is_normalized(q));
	}else
	{
		//printf("fabs(s - 1) < TOLERANCE)");
	}
}


__device__ void quaternion_increment(float* q, const float* rotation, float epsilon_fl) {
	//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_increment() ERROR!"); // Replace assert(quaternion_is_normalized(q))
	float q_old[4] = { q[0],q[1],q[2],q[3] };
	angle_to_quaternion(q, rotation, epsilon_fl);
	angle_to_quaternion_multi_update(q, q_old);
	quaternion_normalize_approx(q, epsilon_fl);
}


__device__ inline flo vec_distance_sqr(const flo* a, const flo* b) {
	flo dx = a[0] - b[0];
    flo dy = a[1] - b[1];
    flo dz = a[2] - b[2];
    
    return dx * dx + dy * dy + dz * dz;
	//return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

__device__  flo gyration_radius(				int				m_lig_begin,
									int				m_lig_end,
						const		atom_cl*		atoms,
						const		m_coords_cl*	m_coords_gpu,
						const		flo*			m_lig_node_origin
) {
	flo acc = 0;
	int counter = 0;
	flo origin[3] = { m_lig_node_origin[0], m_lig_node_origin[1], m_lig_node_origin[2] };
	for (int i = m_lig_begin; i < m_lig_end; i++) {
		flo current_coords[3] = { m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2] };
		if (atoms[i].types[0] != EL_TYPE_H) { // for el, we use the first element (atoms[i].types[0])
			acc += vec_distance_sqr(current_coords, origin);
			++counter;
		}
	}
	return (counter > 0) ? sqrt(acc / counter) : 0;
}

__device__  flo gyration_radius_update(				int				m_lig_begin,
									int				m_lig_end,
								atom_cl*			atoms,
						const		m_coords_cl*	m_coords_gpu,
						const		flo*			m_lig_node_origin,
						const		int				threadNumInBlock,
						const		int				threadsPerBlock
) {
	__shared__ float acc[MAX_NUM_OF_ATOMS];
	//__shared__ float current_coords[MAX_NUM_OF_ATOMS][3];
	__shared__ int counter [MAX_NUM_OF_ATOMS];
	__shared__ flo origin[3];
	origin[0] = m_lig_node_origin[0];
	origin[1] =  m_lig_node_origin[1];
	origin[2] = m_lig_node_origin[2];

	for (int i = threadNumInBlock;
		i < m_lig_end;
		i = i + threadsPerBlock
		)
	{
		counter[i] = 0;
		acc[i] = 0.0;
		//current_coords[i][0] = 0.0;
		//current_coords[i][1] = 0.0;
		//current_coords[i][2] = 0.0;
	}
	//flo acc = 0;
	for (int i = threadNumInBlock; i < m_lig_end; i = i + threadsPerBlock) {
		flo current_coords[3] = { m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2] };
		if (atoms[i].types[0] != EL_TYPE_H) { // for el, we use the first element (atoms[i].types[0])
			acc[i] = vec_distance_sqr(current_coords, origin);
			counter[i] = 1;
		}
	}
	//__threadfence();
	__syncthreads();
	
	if (MAX_NUM_OF_ATOMS > threadsPerBlock) {
		int tmp = MAX_NUM_OF_ATOMS / threadsPerBlock;
		for (int i = 1; i < tmp; i++) {
			acc[threadNumInBlock] += acc[threadNumInBlock + threadsPerBlock * i];
			counter[threadNumInBlock] += counter[threadNumInBlock + threadsPerBlock * i];
		}
		if (threadNumInBlock < (MAX_NUM_OF_ATOMS % threadsPerBlock)) {
			acc[threadNumInBlock] += acc[threadNumInBlock + tmp * threadsPerBlock];
			counter[threadNumInBlock] += counter[threadNumInBlock + tmp * threadsPerBlock];
		}
		//__threadfence();
		__syncthreads();
		for (int off = threadsPerBlock >> 1; off > 0; off >>= 1)
		{
			if (threadNumInBlock < off)
			{
				acc[threadNumInBlock] += acc[threadNumInBlock + off];
				counter[threadNumInBlock] += counter[threadNumInBlock + off];
				
			}
			__syncthreads();
			//__threadfence();
		
		
		}
		return (counter[0] > 0) ? sqrt(acc[0] / counter[0]) : 0;
	}
	else {
		for (int off = (MAX_NUM_OF_ATOMS) >> 1; off > 0; off >>= 1)
		{
			if (threadNumInBlock < off)
			{
				acc[threadNumInBlock] += acc[threadNumInBlock + off];
				counter[threadNumInBlock] += counter[threadNumInBlock + off];

			}
			//__threadfence();
			__syncthreads();
			
		}
		return (counter[0] > 0) ? sqrt(acc[0] / counter[0]) : 0;
	}
}

__device__ void mutate_conf_cl(const		int				step,
											output_type_cl*	c,
										    int*			random_int_map, // delete the  __constant
											flo			random_inside_sphere_map[][3], // delete the  __constant
											flo*			random_fl_pi_map, // delete the  __constant
					const					int				m_lig_begin,
					const					int				m_lig_end,
					const					atom_cl*		atoms,
					const					m_coords_cl*	m_coords_gpu,
					const					flo*			m_lig_node_origin_gpu,
					const					flo			epsilon_fl,
					const					flo			amplitude,
					const					int				lig_torsion_size
) {

	int index = step; // global index (among all threads)
	int which = random_int_map[index];
	int flex_torsion_size = 0;  
		if (which == 0) {
			for (int i = 0; i < 3; i++)
				c->position[i] += amplitude * random_inside_sphere_map[index][i];
			return;
		}
		--which;
		if (which == 0) {
			flo gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			if (gr > epsilon_fl) {
				flo rotation[3];
				for (int i = 0; i < 3; i++)rotation[i] = amplitude / gr * random_inside_sphere_map[index][i];
				quaternion_increment(c->orientation, rotation, epsilon_fl);
			}
			return;
		}
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map[index]; return; }
		which -= lig_torsion_size;

	if (flex_torsion_size != 0) {
		if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map[index]; return; }
		which -= flex_torsion_size;
	}
}

__device__ void mutate_conf_cl_update(const		int				step,
											output_type_cl*	c,
											int*			random_int_map, // delete the  __constant
											flo			random_inside_sphere_map[][3], // delete the  __constant
											flo*			random_fl_pi_map, // delete the  __constant
					const					int				m_lig_begin,
					const					int				m_lig_end,
											atom_cl*		atoms,
					const					m_coords_cl*	m_coords_gpu,
										flo*			m_lig_node_origin_gpu,
					const					flo			epsilon_fl,
					const					flo			amplitude,
					const					int				lig_torsion_size,
					const					int				threadNumInBlock,
					const					int				threadsPerBlock
) {

		int index = step; // global index (among all threads)
		int which = random_int_map[index];
		int flex_torsion_size = 0;  
		if (which == 0) {
			for (int i = threadNumInBlock; i < 3; i += threadsPerBlock)
				c->position[i] += amplitude * random_inside_sphere_map[index][i];
			return;
		}
		--which;
		if (which == 0) {
			flo gr = gyration_radius_update(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu,threadNumInBlock,threadsPerBlock);
			//flo gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			
			if (gr > epsilon_fl) {
				__shared__ flo rotation[3];
				for (int i = threadNumInBlock; i < 3; i += threadsPerBlock)
				{
					rotation[i] = amplitude / gr * random_inside_sphere_map[index][i];
				}
				    __syncthreads();	
					quaternion_increment(c->orientation, rotation, epsilon_fl);		
			}
			return;
		}
				
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map[index]; return; }
		which -= lig_torsion_size;

		if (flex_torsion_size != 0) {
			if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map[index]; return; }
			which -= flex_torsion_size;
		}
	
}
#endif