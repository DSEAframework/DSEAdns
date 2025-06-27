// Data Streaming for Explicit Algorithms - DSEA

#include <dsea.h>
#include <stdio.h>		// printf
#include <cub/cub.cuh>
#include <climits>       // for INT_MAX
#include <fstream>
#include <boost/filesystem.hpp>

using namespace :: std;

// print variadic template values
// overload
template<typename T>
void myprint(T head)
{
    std::cout << head << std::endl;
}
// base case: used when pack is non-empty
template<typename T, typename... Ts>
void myprint(T head, Ts... tail)
{
    std::cout << head << std::endl;
    myprint(tail...);
}

// Calculate the Z Y Coordinates in the grid from an thread ID
// ID: Array Index/Thread ID
// Y,Z: Grid coordinates in part, Y row, Z column
// NC: Number of Columns (NZ)

#define COORDS(ID, Y, Z, NC) \
	do { \
    Y = (ID) / (NC); \
    Z = (ID) % (NC); \
  } while(0)

// Calculate the array index from the grid coordinates
// ID: Array Index/Thread ID
// Y,Z: Grid coordinates in part, Z column, Y row
// NC: Number of Columns (NZ)
#define IDX(Y, Z, ID, NC) \ 
	do { \
    ID = (Y) * (NC) + (Z); \
  } while(0)


// Calculates the global array index from 
__device__ int32_t thread_to_global_idx(int32_t problemsize, int32_t thread_idx, 
						int32_t block_size_z, int32_t block_size_y, 
						int32_t warp_size_z, int32_t warp_size_y,
						int32_t* c_i_block_out, int32_t* r_i_block_out) {
	int32_t global_idx;


	int32_t block_size = block_size_z * block_size_y;
	int32_t num_blocks = (problemsize*problemsize) / block_size;
	int32_t blocks_per_row = (problemsize) / block_size_z;

	int32_t warp_size = warp_size_z * warp_size_y;
	int32_t num_warps_per_block = block_size / warp_size;
	int32_t warps_per_row = block_size_z / warp_size_z;

	int32_t block_idx = thread_idx / block_size;
	int32_t block_row = block_idx / blocks_per_row;
	int32_t block_col = block_idx % blocks_per_row;

	int32_t idx_in_block = thread_idx % block_size;

	int32_t warp_idx = idx_in_block / warp_size;
	int32_t warp_row = warp_idx / warps_per_row;
	int32_t warp_col = warp_idx % warps_per_row;

	int32_t idx_in_warp = idx_in_block % warp_size;

	int32_t col_in_warp = idx_in_warp % warp_size_z;
	int32_t row_in_warp = idx_in_warp / warp_size_z;

	int32_t col_in_block = warp_col * warp_size_z + col_in_warp;
	int32_t row_in_block = warp_row * warp_size_y + row_in_warp;

	int32_t global_row = block_row * block_size_y + row_in_block;
	int32_t global_col = block_col * block_size_z + col_in_block;

	global_idx = global_row * problemsize + global_col;

	*c_i_block_out = col_in_block;
	*r_i_block_out = row_in_block;

	return global_idx;
	

}

__device__ double dns_pDer1(double v_ll, double v_l, double v_r, double v_rr, double DK) {
	return 1./DK * (1./12. * v_ll - 2./3. * v_l + 2./3. * v_r - 1./12. * v_rr);
}

__device__ double dns_pDer2(double v_ll, double v_l, double v_c, double v_r, double v_rr, double DK) {
	return 1./(DK*DK) * (-1./12. * v_ll + 4./3. * v_l - 5./2. * v_c + 4./3. * v_r - 1./12. * v_rr);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// First Partial Derivatives
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dns_dfdx(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f_l, double * __restrict__ f_r,
	/*order 2*/ double * __restrict__ f_ll, double * __restrict__ f_rr,
	/*order 0*/ double * __restrict__ dfdx, int sy_bc_ll = 1, int sy_bc_l = 1, int sy_bc_r = 1, int sy_bc_rr = 1) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		dfdx[idx] = 1/DX * (1./12. * sy_bc_ll *  f_ll[idx] - 2./3. * sy_bc_l * f_l[idx] + 2./3. * sy_bc_r * f_r[idx] - 1./12. * sy_bc_rr * f_rr[idx]); 
		//printf("dfdx My Id: %d, values: %lf/%lf/%lf/%lf\n", idx, f_ll[idx], f_l[idx], f_r[idx], f_rr[idx]);
	}

}

__global__ void dns_dfgdx(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f_l, double * __restrict__ f_r,
	/*order 2*/ double * __restrict__ f_ll, double * __restrict__ f_rr,
	/*order 1*/ double * __restrict__ g_l, double * __restrict__ g_r,
	/*order 2*/ double * __restrict__ g_ll, double * __restrict__ g_rr,
	/*order 0*/ double * __restrict__ dfgdx, 
	int sy_bc_f_ll = 1, int sy_bc_f_l = 1, int sy_bc_f_r = 1, int sy_bc_f_rr = 1,
	int sy_bc_g_ll = 1, int sy_bc_g_l = 1, int sy_bc_g_r = 1, int sy_bc_g_rr = 1) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		dfgdx[idx] = 1/DX * (1./12. * (sy_bc_f_ll * f_ll[idx] * sy_bc_g_ll * g_ll[idx]) - 2./3. * (sy_bc_f_l * f_l[idx] * sy_bc_g_l * g_l[idx]) 
							+ 2./3. * (sy_bc_f_r * f_r[idx] * sy_bc_g_r * g_r[idx]) - 1./12. * (sy_bc_f_rr * f_rr[idx] * sy_bc_g_rr * g_rr[idx])); 
	}

}

__global__ void dns_dfdyz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfdy, 
	/*order 0*/ double * __restrict__ dfdz) {

	__shared__ double s_d[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo
	
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (gidx<block_ncc) {
		row_in_block+=2;
		col_in_block+=2;

		s_d[row_in_block][col_in_block] = f[gidx];

		int32_t Y, Z;
		int32_t dy_ll, dy_rr, dz_ll, dz_rr;

		COORDS(gidx, Y, Z, NZ);

		// get halos
		if (row_in_block < 4) {
			IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
			s_d[row_in_block-2][col_in_block] = f[dy_ll];
		}
		if (row_in_block >= BLOCKSIZE_Y) {
			IDX((NY+Y+2)%NY, Z, dy_rr, NZ);
			s_d[row_in_block+2][col_in_block] = f[dy_rr];
		}

		// get halos
		if (col_in_block < 4) {
			IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
			s_d[row_in_block][col_in_block-2] = f[dz_ll];
		}
		if (col_in_block >= BLOCKSIZE_Z) {
			IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);
			s_d[row_in_block][col_in_block+2] = f[dz_rr];
		}

		__syncthreads();

		dfdy[gidx] = 1/DY * (1./12. * s_d[row_in_block-2][col_in_block] - 2./3. * s_d[row_in_block-1][col_in_block] 
			+ 2./3. * s_d[row_in_block+1][col_in_block]  - 1./12. * s_d[row_in_block+2][col_in_block]); 

		dfdz[gidx] = 1/DZ * (1./12. * s_d[row_in_block][col_in_block-2] - 2./3. * s_d[row_in_block][col_in_block-1] 
			+ 2./3. * s_d[row_in_block][col_in_block+1] - 1./12. * s_d[row_in_block][col_in_block+2]); 
	}
}

__global__ void dns_dfgdyz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 1*/ double * __restrict__ g,
	/*order 0*/ double * __restrict__ dfgdy,
	/*order 0*/ double * __restrict__ dfgdz) {

	__shared__ double s_d[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);
	
	if (gidx<block_ncc) {
		row_in_block+=2;
		col_in_block+=2;

		s_d[row_in_block][col_in_block] = f[gidx] * g[gidx];

		int32_t Y, Z;
		int32_t dy_ll, dy_rr, dz_ll, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// get halos
		if (row_in_block < 4) {
			IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
			s_d[row_in_block-2][col_in_block] = f[dy_ll] * g[dy_ll];
		}
		if (row_in_block >= BLOCKSIZE_Y) {
			IDX((NY+Y+2)%NY, Z, dy_rr, NZ);
			s_d[row_in_block+2][col_in_block] = f[dy_rr] * g[dy_rr];
		}

		// get halos
		if (col_in_block < 4) {
			IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
			s_d[row_in_block][col_in_block-2] = f[dz_ll] * g[dz_ll];
		}
		if (col_in_block >= BLOCKSIZE_Z) {
			IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);
			s_d[row_in_block][col_in_block+2] = f[dz_rr] * g[dz_rr];
		}

		__syncthreads();

		dfgdy[gidx] = 1/DY * (1./12. * s_d[row_in_block-2][col_in_block] - 2./3. * s_d[row_in_block-1][col_in_block] 
			+ 2./3. * s_d[row_in_block+1][col_in_block]  - 1./12. * s_d[row_in_block+2][col_in_block]); 
		
		dfgdz[gidx] = 1/DZ * (1./12. * s_d[row_in_block][col_in_block-2] - 2./3. * s_d[row_in_block][col_in_block-1]
			+ 2./3. * s_d[row_in_block][col_in_block+1] - 1./12. * s_d[row_in_block][col_in_block+2]); 


	}
}


__global__ void dns_dfdy(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfdy) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, idx_ll, NZ);
		IDX((NY+Y-1)%NY, Z, idx_l, NZ);
		IDX((NY+Y+1)%NY, Z, idx_r, NZ);
		IDX((NY+Y+2)%NY, Z, idx_rr, NZ);

		dfdy[idx] = 1/DY * (1./12. * f[idx_ll] - 2./3. * f[idx_l] + 2./3. * f[idx_r] - 1./12. * f[idx_rr]); 

		//printf("dfdy My Id: %d, my slice coordinates %d,%d, neighbors: %d/%d/%d/%d, values: %lf/%lf/%lf/%lf/%lf, result %lf\n", idx, Y,Z, idx_ll, idx_l, idx_r, idx_rr, f[idx_ll], f[idx_l], f[idx], f[idx_r], f[idx_rr], dfdy[idx]);
	}
}

__global__ void dns_dfgdy(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 1*/ double * __restrict__ g,
	/*order 0*/ double * __restrict__ dfgdy) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);
	
	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, idx_ll, NZ);
		IDX((NY+Y-1)%NY, Z, idx_l, NZ);
		IDX((NY+Y+1)%NY, Z, idx_r, NZ);
		IDX((NY+Y+2)%NY, Z, idx_rr, NZ);

		dfgdy[idx] = 1/DY * (1./12. * (f[idx_ll] * g[idx_ll]) - 2./3. * (f[idx_l] * g[idx_l]) 
								+ 2./3. * (f[idx_r] * g[idx_r]) - 1./12. * (f[idx_rr] * g[idx_rr])); 
	}
}

__global__ void dns_dfdz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfdz) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, idx_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, idx_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, idx_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, idx_rr, NZ);

		dfdz[idx] = 1/DZ * (1./12. * f[idx_ll] - 2./3. * f[idx_l] + 2./3. * f[idx_r] - 1./12. * f[idx_rr]); 

		//printf("dfdz My Id: %d, my slice coordinates %d,%d, neighbors: %d/%d/%d/%d\n", idx, Y,Z, idx_ll, idx_l, idx_r, idx_rr);
	}

}

__global__ void dns_dfgdz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 1*/ double * __restrict__ g,
	/*order 0*/ double * __restrict__ dfgdz) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, idx_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, idx_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, idx_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, idx_rr, NZ);

		dfgdz[idx] = 1/DZ * (1./12. * (f[idx_ll] * g[idx_ll]) - 2./3. * (f[idx_l] * g[idx_l]) 
							+ 2./3. * (f[idx_r] * g[idx_r]) - 1./12. * (f[idx_rr] * g[idx_rr])); 
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Second Partial Derivatives
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dns_dfd2x(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f_c,
	/*order 1*/ double * __restrict__ f_l, double * __restrict__ f_r,
	/*order 2*/ double * __restrict__ f_ll, double * __restrict__ f_rr,
	/*order 0*/ double * __restrict__ dfd2x, int sy_bc_ll = 1, int sy_bc_l = 1, int sy_bc_r = 1, int sy_bc_rr = 1) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		dfd2x[idx] = 1/(DX*DX) * (-1./12. * sy_bc_ll * f_ll[idx] + 4./3. * sy_bc_l * f_l[idx] - 5./2. * f_c[idx] + 4./3. * sy_bc_r * f_r[idx] - 1./12. * sy_bc_rr * f_rr[idx]); 
	}

}

__global__ void dns_dfd2yz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfd2y,
	/*order 0*/ double * __restrict__ dfd2z) {

	__shared__ double s_d[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);
	
	if (gidx<block_ncc) {
		row_in_block+=2;
		col_in_block+=2;

		s_d[row_in_block][col_in_block] = f[gidx];

		int32_t Y, Z;
		int32_t dy_ll, dy_rr, dz_ll, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// get halos
		if (row_in_block < 4) {
			IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
			s_d[row_in_block-2][col_in_block] = f[dy_ll];
		}
		if (row_in_block >= BLOCKSIZE_Y) {
			IDX((NY+Y+2)%NY, Z, dy_rr, NZ);
			s_d[row_in_block+2][col_in_block] = f[dy_rr];
		}

		// get halos
		if (col_in_block < 4) {
			IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
			s_d[row_in_block][col_in_block-2] = f[dz_ll];
		}
		if (col_in_block >= BLOCKSIZE_Z) {
			IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);
			s_d[row_in_block][col_in_block+2] = f[dz_rr];
		}

		__syncthreads();

		dfd2y[gidx] = 1/(DY*DY) * (-1./12. * s_d[row_in_block-2][col_in_block] + 4./3. * s_d[row_in_block-1][col_in_block] 
			- 5./2. * s_d[row_in_block][col_in_block] 
			+ 4./3. * s_d[row_in_block+1][col_in_block] - 1./12. * s_d[row_in_block+2][col_in_block]); 

		dfd2z[gidx] = 1/(DZ*DZ) * (-1./12. * s_d[row_in_block][col_in_block-2] + 4./3. * s_d[row_in_block][col_in_block-1] 
			- 5./2. * s_d[row_in_block][col_in_block] 
			+ 4./3. * s_d[row_in_block][col_in_block+1] - 1./12. * s_d[row_in_block][col_in_block+2]); 
	}
}

__global__ void dns_dfd2y(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 1*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfd2y) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);
	
	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, idx_ll, NZ);
		IDX((NY+Y-1)%NY, Z, idx_l, NZ);
		IDX((NY+Y+1)%NY, Z, idx_r, NZ);
		IDX((NY+Y+2)%NY, Z, idx_rr, NZ);

		//printf("dfd2y My Id: %d, my slice coordinates %d,%d, neighbors: %d/%d/%d/%d\n", idx, Y,Z, idx_ll, idx_l, idx_r, idx_rr);

		dfd2y[idx] = 1/(DY*DY) * (-1./12. * f[idx_ll] + 4./3. * f[idx_l] - 5./2. * f[idx] + 4./3. * f[idx_r] - 1./12. * f[idx_rr]); 

	}
}

__global__ void dns_dfd2z(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ dfd2z) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);
	
	if (idx<block_ncc) {
		int32_t Y, Z;
		int32_t idx_ll, idx_l, idx_r, idx_rr;
		COORDS(idx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, idx_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, idx_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, idx_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, idx_rr, NZ);

		//printf("dfd2z My Id: %d, my slice coordinates %d,%d, neighbors: %d/%d/%d/%d\n", idx, Y,Z, idx_ll, idx_l, idx_r, idx_rr);

		dfd2z[idx] = 1/(DZ*DZ) * (-1./12. * f[idx_ll] + 4./3. * f[idx_l] - 5./2. * f[idx] + 4./3. * f[idx_r] - 1./12. * f[idx_rr]); 
	}
}


__global__ void dns_Res_v1(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ rho,
	/*order 0*/ double * __restrict__ u0,
	/*order 0*/ double * __restrict__ u1,
	/*order 0*/ double * __restrict__ u2,
	/*order 0*/ double * __restrict__ rhou0,
	/*order 0*/ double * __restrict__ rhou1,
	/*order 0*/ double * __restrict__ rhou2,
	/*order 0*/ double * __restrict__ rhoE,
	/*order 0*/ double * __restrict__ p,
	/*order 0*/ double * __restrict__ T,

	/*order 0*/ double * __restrict__ du0dx,
	/*order 0*/ double * __restrict__ du0dy,
	/*order 0*/ double * __restrict__ du0dz,

	/*order 0*/ double * __restrict__ du1dx,
	/*order 0*/ double * __restrict__ du1dy,
	/*order 0*/ double * __restrict__ du1dz,

	/*order 0*/ double * __restrict__ du2dx,
	/*order 0*/ double * __restrict__ du2dy,
	/*order 0*/ double * __restrict__ du2dz,

	/*order 0*/ double * __restrict__ drhodx,
	/*order 0*/ double * __restrict__ drhody,
	/*order 0*/ double * __restrict__ drhodz,

	/*order 0*/ double * __restrict__ drhou0dx,
	/*order 0*/ double * __restrict__ drhou0dy,
	/*order 0*/ double * __restrict__ drhou0dz,

	/*order 0*/ double * __restrict__ drhou1dx,
	/*order 0*/ double * __restrict__ drhou1dy,
	/*order 0*/ double * __restrict__ drhou1dz,

	/*order 0*/ double * __restrict__ drhou2dx,
	/*order 0*/ double * __restrict__ drhou2dy,
	/*order 0*/ double * __restrict__ drhou2dz,

	/*order 0*/ double * __restrict__ dpdx,
	/*order 0*/ double * __restrict__ dpdy,
	/*order 0*/ double * __restrict__ dpdz,

	/*order 0*/ double * __restrict__ dpu0dx,
	/*order 0*/ double * __restrict__ dpu1dy,
	/*order 0*/ double * __restrict__ dpu2dz,

	/*order 0*/ double * __restrict__ drhou0u0dx,
	/*order 0*/ double * __restrict__ drhou0u1dy,
	/*order 0*/ double * __restrict__ drhou0u2dz,

	/*order 0*/ double * __restrict__ drhou1u0dx,
	/*order 0*/ double * __restrict__ drhou1u1dy,
	/*order 0*/ double * __restrict__ drhou1u2dz,

	/*order 0*/ double * __restrict__ drhou2u0dx,
	/*order 0*/ double * __restrict__ drhou2u1dy,
	/*order 0*/ double * __restrict__ drhou2u2dz,

	/*order 0*/ double * __restrict__ du0d2x,
	/*order 0*/ double * __restrict__ du0d2y,
	/*order 0*/ double * __restrict__ du0d2z,

	/*order 0*/ double * __restrict__ du1d2x,
	/*order 0*/ double * __restrict__ du1d2y,
	/*order 0*/ double * __restrict__ du1d2z,

	/*order 0*/ double * __restrict__ du2d2x,
	/*order 0*/ double * __restrict__ du2d2y,
	/*order 0*/ double * __restrict__ du2d2z,

	/*order 0*/ double * __restrict__ du0dxdy,
	/*order 0*/ double * __restrict__ du0dxdz,

	/*order 0*/ double * __restrict__ du1dxdy,
	/*order 0*/ double * __restrict__ du1dydz,

	/*order 0*/ double * __restrict__ du2dxdz,
	/*order 0*/ double * __restrict__ du2dydz,

	/*order 0*/ double * __restrict__ dTd2x,
	/*order 0*/ double * __restrict__ dTd2y,
	/*order 0*/ double * __restrict__ dTd2z,

	/*order 0*/ double * __restrict__ drhoEdx,
	/*order 0*/ double * __restrict__ drhoEdy,
	/*order 0*/ double * __restrict__ drhoEdz,

	/*order 0*/ double * __restrict__ drhoEu0dx,
	/*order 0*/ double * __restrict__ drhoEu1dy,
	/*order 0*/ double * __restrict__ drhoEu2dz,
	
	/*order 0*/ double * __restrict__ Res_rho,
	/*order 0*/ double * __restrict__ Res_rhou0,
	/*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ Res_rhou2,
	/*order 0*/ double * __restrict__ Res_rhoE) {


	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {

		double lRes_rho = 0;
		double lRes_rhou0 = 0;
		double lRes_rhou1 = 0;
		double lRes_rhou2 = 0;
		double lRes_rhoE = 0;

		double ldu0dx = du0dx[idx];
		double ldu1dy = du1dy[idx];
		double ldu2dz = du2dz[idx];

		double tmp0 = -0.5 * (ldu0dx + ldu1dy + ldu2dz);

		lRes_rho += tmp0 * rho[idx];
		lRes_rhou0 += tmp0 * rhou0[idx];
		lRes_rhou1 += tmp0 * rhou1[idx];
		lRes_rhou2 += tmp0 * rhou2[idx];
		lRes_rhoE += tmp0 * rhoE[idx];

		double frac0 = 1./RE;
		double frac1 = 2./3.;
		double frac2 = 4./3.;
		double frac3 = 1./3.;
		lRes_rhoE += frac0 * (-frac1 * ldu0dx - frac1 * ldu1dy + frac2 * ldu2dz) * ldu2dz;
		lRes_rhoE += frac0 * (-frac1 * ldu0dx + frac2 * ldu1dy - frac1 * ldu2dz) * ldu1dy;
		lRes_rhoE += frac0 * ( frac2 * ldu0dx - frac1 * ldu1dy - frac1 * ldu2dz) * ldu0dx;

		double lu0 = u0[idx];
		lRes_rho +=   -0.5 * lu0 * drhodx[idx];
		lRes_rhou1 += -0.5 * lu0 * drhou1dx[idx];
		lRes_rhou2 += -0.5 * lu0 * drhou2dx[idx];
		lRes_rhoE +=  -0.5 * drhoEdx[idx] * lu0;

		double ldrhou0dx = drhou0dx[idx];
		lRes_rhou0 += -0.5 * lu0 * ldrhou0dx;
		lRes_rho +=   -0.5 * ldrhou0dx;

		tmp0 = frac0 * (frac2 * du0d2x[idx] + du0d2y[idx] + du0d2z[idx]
			+ frac3 * du1dxdy[idx]
			+ frac3 * du2dxdz[idx]);

		lRes_rhou0 += tmp0;
		lRes_rhoE += lu0 * tmp0;



		double lu1 = u1[idx];
		lRes_rho +=   -0.5 * lu1 * drhody[idx];
		lRes_rhou0 += -0.5 * lu1 * drhou0dy[idx];
		lRes_rhou2 += -0.5 * lu1 * drhou2dy[idx];
		lRes_rhoE +=  -0.5 * drhoEdy[idx] * lu1;

		double ldrhou1dy = drhou1dy[idx];
		lRes_rhou1 += -0.5 * lu1 * ldrhou1dy;
		lRes_rho +=   -0.5 * ldrhou1dy;

		tmp0 = frac0 * (frac2 * du1d2y[idx] + du1d2x[idx] + du1d2z[idx]
			+ frac3 * du0dxdy[idx]
			+ frac3 * du2dydz[idx]);

		lRes_rhou1 += tmp0;
		lRes_rhoE += lu1 * tmp0;



		double lu2 = u2[idx];
		lRes_rho +=   -0.5 * lu2 * drhodz[idx];
		lRes_rhou0 += -0.5 * lu2 * drhou0dz[idx];
		lRes_rhou1 += -0.5 * lu2 * drhou1dz[idx];
		lRes_rhoE +=  -0.5 * drhoEdz[idx] * lu2;

		double ldrhou2dz = drhou2dz[idx];
		lRes_rhou2 += -0.5 * lu2 * ldrhou2dz;
		lRes_rho +=   -0.5 * ldrhou2dz;

		Res_rho[idx] = lRes_rho;

		tmp0 = frac0 * (frac2 * du2d2z[idx] + du2d2x[idx] + du2d2y[idx]
			+ frac3 * du0dxdz[idx]
			+ frac3 * du1dydz[idx]);

		lRes_rhou2 += tmp0;
		lRes_rhoE += lu2 * tmp0;


		lRes_rhou0 += -0.5 * drhou0u0dx[idx] - 0.5 * drhou0u1dy[idx] - 0.5 * drhou0u2dz[idx];
		lRes_rhou0 += -dpdx[idx];

		Res_rhou0[idx] = lRes_rhou0;


		lRes_rhou1 += -0.5 * drhou1u0dx[idx] - 0.5 * drhou1u1dy[idx] - 0.5 * drhou1u2dz[idx];
		lRes_rhou1 += -dpdy[idx];

		Res_rhou1[idx] = lRes_rhou1;


		lRes_rhou2 += -0.5 * drhou2u0dx[idx] - 0.5 * drhou2u1dy[idx] - 0.5 * drhou2u2dz[idx];
		lRes_rhou2 += -dpdz[idx];

		Res_rhou2[idx] = lRes_rhou2;


		lRes_rhoE += -dpu0dx[idx] - dpu1dy[idx] - dpu2dz[idx];
		lRes_rhoE += -0.5 * drhoEu0dx[idx] - 0.5 * drhoEu1dy[idx] - 0.5 * drhoEu2dz[idx];


		lRes_rhoE += 1./RE *	(du0dy[idx] + du1dx[idx]) * du0dy[idx]
					+ 1./RE *	(du0dy[idx] + du1dx[idx]) * du1dx[idx];

		lRes_rhoE += 1./RE *	(du0dz[idx] + du2dx[idx]) * du0dz[idx]
					+ 1./RE *	(du0dz[idx] + du2dx[idx]) * du2dx[idx];

		lRes_rhoE += 1./RE *	(du1dz[idx] + du2dy[idx]) * du1dz[idx]
					+ 1./RE *	(du1dz[idx] + du2dy[idx]) * du2dy[idx];

		lRes_rhoE += (dTd2x[idx] + dTd2y[idx] + dTd2z[idx]) / (MINF * MINF * PR * RE * (GAMA - 1));

		Res_rhoE[idx] = lRes_rhoE;

	}



}

__global__ void dns_Res(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ rho,
	/*order 0*/ double * __restrict__ u0,
	/*order 0*/ double * __restrict__ u1,
	/*order 0*/ double * __restrict__ u2,
	/*order 0*/ double * __restrict__ rhou0,
	/*order 0*/ double * __restrict__ rhou1,
	/*order 0*/ double * __restrict__ rhou2,
	/*order 0*/ double * __restrict__ rhoE,
	/*order 0*/ double * __restrict__ p,
	/*order 0*/ double * __restrict__ T,

	/*order 0*/ double * __restrict__ du0dx,
	/*order 0*/ double * __restrict__ du0dy,
	/*order 0*/ double * __restrict__ du0dz,

	/*order 0*/ double * __restrict__ du1dx,
	/*order 0*/ double * __restrict__ du1dy,
	/*order 0*/ double * __restrict__ du1dz,

	/*order 0*/ double * __restrict__ du2dx,
	/*order 0*/ double * __restrict__ du2dy,
	/*order 0*/ double * __restrict__ du2dz,

	/*order 0*/ double * __restrict__ drhodx,
	/*order 0*/ double * __restrict__ drhody,
	/*order 0*/ double * __restrict__ drhodz,

	/*order 0*/ double * __restrict__ drhou0dx,
	/*order 0*/ double * __restrict__ drhou0dy,
	/*order 0*/ double * __restrict__ drhou0dz,

	/*order 0*/ double * __restrict__ drhou1dx,
	/*order 0*/ double * __restrict__ drhou1dy,
	/*order 0*/ double * __restrict__ drhou1dz,

	/*order 0*/ double * __restrict__ drhou2dx,
	/*order 0*/ double * __restrict__ drhou2dy,
	/*order 0*/ double * __restrict__ drhou2dz,

	/*order 0*/ double * __restrict__ dpdx,
	/*order 0*/ double * __restrict__ dpdy,
	/*order 0*/ double * __restrict__ dpdz,

	/*order 0*/ double * __restrict__ dpu0dx,
	/*order 0*/ double * __restrict__ dpu1dy,
	/*order 0*/ double * __restrict__ dpu2dz,

	/*order 0*/ double * __restrict__ drhou0u0dx,
	/*order 0*/ double * __restrict__ drhou0u1dy,
	/*order 0*/ double * __restrict__ drhou0u2dz,

	/*order 0*/ double * __restrict__ drhou1u0dx,
	/*order 0*/ double * __restrict__ drhou1u1dy,
	/*order 0*/ double * __restrict__ drhou1u2dz,

	/*order 0*/ double * __restrict__ drhou2u0dx,
	/*order 0*/ double * __restrict__ drhou2u1dy,
	/*order 0*/ double * __restrict__ drhou2u2dz,

	/*order 0*/ double * __restrict__ du0d2x,
	/*order 0*/ double * __restrict__ du0d2y,
	/*order 0*/ double * __restrict__ du0d2z,

	/*order 0*/ double * __restrict__ du1d2x,
	/*order 0*/ double * __restrict__ du1d2y,
	/*order 0*/ double * __restrict__ du1d2z,

	/*order 0*/ double * __restrict__ du2d2x,
	/*order 0*/ double * __restrict__ du2d2y,
	/*order 0*/ double * __restrict__ du2d2z,

	/*order 0*/ double * __restrict__ du0dxdy,
	/*order 0*/ double * __restrict__ du0dxdz,

	/*order 0*/ double * __restrict__ du1dxdy,
	/*order 0*/ double * __restrict__ du1dydz,

	/*order 0*/ double * __restrict__ du2dxdz,
	/*order 0*/ double * __restrict__ du2dydz,

	/*order 0*/ double * __restrict__ dTd2x,
	/*order 0*/ double * __restrict__ dTd2y,
	/*order 0*/ double * __restrict__ dTd2z,

	/*order 0*/ double * __restrict__ drhoEdx,
	/*order 0*/ double * __restrict__ drhoEdy,
	/*order 0*/ double * __restrict__ drhoEdz,

	/*order 0*/ double * __restrict__ drhoEu0dx,
	/*order 0*/ double * __restrict__ drhoEu1dy,
	/*order 0*/ double * __restrict__ drhoEu2dz,
	
	/*order 0*/ double * __restrict__ Res_rho,
	/*order 0*/ double * __restrict__ Res_rhou0,
	/*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ Res_rhou2,
	/*order 0*/ double * __restrict__ Res_rhoE) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (idx<block_ncc) {
		/*
		Res_rho[idx] = 0.5 * (- rho[idx] * (du0dx[idx] + du1dy[idx] + du2dz[idx]) 
			- drhodx[idx] * u0[idx] - drhody[idx] * u1[idx] - drhodz[idx] * u2[idx] 
			- drhou0dx[idx] - drhou1dy[idx] - drhou2dz[idx]);
		*/
		Res_rho[idx] = -0.5 * (du1dy[idx] + du2dz[idx] + du0dx[idx]) * rho[idx] 
						- 0.5 * u0[idx] * drhodx[idx] - 0.5 * u1[idx] * drhody[idx] - 0.5 * u2[idx] * drhodz[idx] 
						- 0.5 * drhou1dy[idx] - 0.5 * drhou2dz[idx] - 0.5 * drhou0dx[idx];


		// -0.5 * (du1/dy + du2/dz + du0dx) * rho - 0.5 * u0 * drhodx - 0.5 * u1 * drhody - 0.5 * u2 * drhodz - 0.5 * drhou1dy - 0.5 * drhou2dz - 0.5 * drhou0dx

		Res_rhou0[idx] = -0.5 * (du0dx[idx] + du1dy[idx] + du2dz[idx]) * rhou0[idx]
							-0.5 * u0[idx] * drhou0dx[idx] - 0.5 * u1[idx] * drhou0dy[idx] - 0.5 * u2[idx] * drhou0dz[idx] 
							-0.5 * drhou0u0dx[idx] - 0.5 * drhou0u1dy[idx] - 0.5 * drhou0u2dz[idx]
							- dpdx[idx]
							+ 1. / RE * (4./3. * du0d2x[idx] + du0d2y[idx] + du0d2z[idx] 
							+ 1./3. * du1dxdy[idx] 
							+ 1./3. * du2dxdz[idx]);

		Res_rhou1[idx] = -0.5 * (du0dx[idx] + du1dy[idx] + du2dz[idx]) * rhou1[idx]
							-0.5 * u0[idx] * drhou1dx[idx] - 0.5 * u1[idx] * drhou1dy[idx] - 0.5 * u2[idx] * drhou1dz[idx] 
							-0.5 * drhou1u0dx[idx] - 0.5 * drhou1u1dy[idx] - 0.5 * drhou1u2dz[idx]
							- dpdy[idx]
							+ 1. / RE * (4./3. * du1d2y[idx] + du1d2x[idx] + du1d2z[idx] 
							+ 1./3. * du0dxdy[idx] 
							+ 1./3. * du2dydz[idx]);

		Res_rhou2[idx] = -0.5 * (du0dx[idx] + du1dy[idx] + du2dz[idx]) * rhou2[idx]
							-0.5 * u0[idx] * drhou2dx[idx] - 0.5 * u1[idx] * drhou2dy[idx] - 0.5 * u2[idx] * drhou2dz[idx] 
							-0.5 * drhou2u0dx[idx] - 0.5 * drhou2u1dy[idx] - 0.5 * drhou2u2dz[idx]
							- dpdz[idx]
							+ 1. / RE * (4./3. * du2d2z[idx] + du2d2x[idx] + du2d2y[idx] 
							+ 1./3. * du0dxdz[idx] 
							+ 1./3. * du1dydz[idx]);

		Res_rhoE[idx] = -0.5 * (du0dx[idx] + du1dy[idx] + du2dz[idx]) * rhoE[idx]
							-0.5 * drhoEdx[idx] * u0[idx] - 0.5 * drhoEdy[idx] * u1[idx] - 0.5 * drhoEdz[idx] * u2[idx]
							-dpu0dx[idx] - dpu1dy[idx] - dpu2dz[idx] 
							-0.5 * drhoEu0dx[idx] - 0.5 * drhoEu1dy[idx] - 0.5 * drhoEu2dz[idx]
							+ u0[idx] / RE * (4./3. * du0d2x[idx] + du0d2y[idx] + du0d2z[idx]
								+ 1./3. * du1dxdy[idx] 
								+ 1./3. * du2dxdz[idx])
							+ u1[idx] / RE * (1./3. * du0dxdy[idx] 
								+ du1d2x[idx] + 4./3. * du1d2y[idx] + du1d2z[idx]
								+ 1./3. * du2dydz[idx])
							+ u2[idx] / RE * (1./3. * du0dxdz[idx]
								+ 1./3. * du1dydz[idx]
								+ du2d2x[idx] + du2d2y[idx] + 4./3. * du2d2z[idx])
							+ 1./RE *	(du0dy[idx] + du1dx[idx]) * du0dy[idx]
							+ 1./RE *	(du0dy[idx] + du1dx[idx]) * du1dx[idx]

							+ 1./RE *	(du0dz[idx] + du2dx[idx]) * du0dz[idx]
							+ 1./RE *	(du0dz[idx] + du2dx[idx]) * du2dx[idx]

							+ 1./RE *	(du1dz[idx] + du2dy[idx]) * du1dz[idx]
							+ 1./RE *	(du1dz[idx] + du2dy[idx]) * du2dy[idx]

							+ 1./RE *	(-2./3. * du0dx[idx] - 2./3. * du1dy[idx] + 4./3. * du2dz[idx]) * du2dz[idx]
							+ 1./RE *	(-2./3. * du0dx[idx] + 4./3. * du1dy[idx] - 2./3. * du2dz[idx]) * du1dy[idx]
							+ 1./RE *	( 4./3. * du0dx[idx] - 2./3. * du1dy[idx] - 2./3. * du2dz[idx]) * du0dx[idx]
						
							+ (dTd2x[idx] + dTd2y[idx] + dTd2z[idx]) / (MINF * MINF * PR * RE * (GAMA - 1));

	}



}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dns_copy(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ g) {
	int32_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<block_ncc) {
		g[idx] = f[idx];
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constituent relations
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dns_fdivg(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f,
	/*order 0*/ double * __restrict__ g,
	/*order 0*/ double * __restrict__ res) {
	int32_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<block_ncc) {
		res[idx] = f[idx] / g[idx];
	}
}

__global__ void dns_p(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ rho,
	/*order 0*/ double * __restrict__ u0,
	/*order 0*/ double * __restrict__ u1,
	/*order 0*/ double * __restrict__ u2,
	/*order 0*/ double * __restrict__ rhoE,
	/*order 0*/ double * __restrict__ p) {
	
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (idx<block_ncc) {
		p[idx] = (GAMA - 1) * (rhoE[idx] - 
				0.5 * rho[idx]  * u0[idx] * u0[idx] -
				0.5 * rho[idx]  * u1[idx] * u1[idx] -
				0.5 * rho[idx]  * u2[idx] * u2[idx]);
	}

}

__global__ void dns_T(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ rho,
	/*order 0*/ double * __restrict__ p,
	/*order 0*/ double * __restrict__ T) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		T[idx] = MINF * MINF * GAMA * p[idx] / rho[idx];
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Time advancement
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dns_RKsubStage(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f0,
	/*order 0*/ double * __restrict__ f1,
	/*order 0*/ double * __restrict__ f2,
	/*order 0*/ double * __restrict__ f3,
	/*order 0*/ double * __restrict__ f4,
	/*order 0*/ double * __restrict__ f0_old,
	/*order 0*/ double * __restrict__ f1_old,
	/*order 0*/ double * __restrict__ f2_old,
	/*order 0*/ double * __restrict__ f3_old,
	/*order 0*/ double * __restrict__ f4_old,
	/*order 0*/ double * __restrict__ Residual0,
	/*order 0*/ double * __restrict__ Residual1,
	/*order 0*/ double * __restrict__ Residual2,
	/*order 0*/ double * __restrict__ Residual3,
	/*order 0*/ double * __restrict__ Residual4,
	/*order 0*/ double rk) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		f0[idx] = DT * rk * Residual0[idx] + f0_old[idx];
		f1[idx] = DT * rk * Residual1[idx] + f1_old[idx];
		f2[idx] = DT * rk * Residual2[idx] + f2_old[idx];
		f3[idx] = DT * rk * Residual3[idx] + f3_old[idx];
		f4[idx] = DT * rk * Residual4[idx] + f4_old[idx];
	}
}

__global__ void dns_RKtmpAdvance(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f0_old_out,
	/*order 0*/ double * __restrict__ f1_old_out,
	/*order 0*/ double * __restrict__ f2_old_out,
	/*order 0*/ double * __restrict__ f3_old_out,
	/*order 0*/ double * __restrict__ f4_old_out,
	/*order 0*/ double * __restrict__ f0_old,
	/*order 0*/ double * __restrict__ f1_old,
	/*order 0*/ double * __restrict__ f2_old,
	/*order 0*/ double * __restrict__ f3_old,
	/*order 0*/ double * __restrict__ f4_old,
	/*order 0*/ double * __restrict__ Residual0,
	/*order 0*/ double * __restrict__ Residual1,
	/*order 0*/ double * __restrict__ Residual2,
	/*order 0*/ double * __restrict__ Residual3,
	/*order 0*/ double * __restrict__ Residual4,
	/*order 0*/ double rk) {

	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;

	int32_t col_in_block, row_in_block;
	int32_t idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (idx<block_ncc) {
		f0_old_out[idx] = DT * rk * Residual0[idx] + f0_old[idx];
		f1_old_out[idx] = DT * rk * Residual1[idx] + f1_old[idx];
		f2_old_out[idx] = DT * rk * Residual2[idx] + f2_old[idx];
		f3_old_out[idx] = DT * rk * Residual3[idx] + f3_old[idx];
		f4_old_out[idx] = DT * rk * Residual4[idx] + f4_old[idx];
	}
}

__global__ void dns_init(const double * __restrict__ p_in, double * __restrict__ p_out) {
	// Calculate position in part
	int32_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<block_header_size) {
		p_out[idx]=p_in[idx];
	}


}


__global__ void dns_DebugAdvance(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f0_out,
	/*order 0*/ double * __restrict__ f0_in) {

	int32_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<block_ncc) {
		f0_out[idx] = f0_in[idx] + 1.0;

	}
}

__global__ void dns_Debug(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ f0_in, /*order 0*/ double * __restrict__ f0_out) {

	int32_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<2) {
		printf("In: f0[%d] = %lf    ", idx, f0_in[idx]);
		if (idx == 0) {
			f0_out[idx] = f0_in[idx];
		}
		if (idx == 1) {
			printf("Out old: f0[%d] = %lf    ", idx, f0_out[idx]);
			f0_out[idx] = f0_in[idx] + 1;
			printf("Out new: f0[%d] = %lf    ", idx, f0_out[idx]);
		}

	}
}



void DS::caller_worker (double ** p_in, double ** p_out, int32_t i_part, int32_t i_super_cycle,
						int32_t order_in, int32_t order_out, int32_t iworker, int32_t nworker,
						cudaStream_t * stream, int32_t threads_per_block, int32_t blockSize, int32_t myID) {

	//cout << "in:" << p_in[0] << " " << p_in[1] << " " << p_in[2] << " " << p_in[3] << " " << p_in[4] << " " << p_in[5] << endl;
	//cout << "out:" << p_out[0] << " " << p_out[1] << " " << p_out[2] << " " << p_out[3] << " " << p_out[4] << " " << p_out[5] << endl;

	// the order of arrays in p_in and p_out is:
	// center, left, right, left-left, right-right, left-left-left, right-right-right, and so on
	// entries can be NULL when invalid

	//cout << NX << ", " << NY << ", " << NZ << ", " << DX << ", " << DY << ", " << DZ <<  ", " <<  DT << ", " <<  GAMA <<  ", " <<  MINF <<  ", " <<  RE <<  ", " <<  PR << endl;


	int32_t global_worker_id = nworker * myID + iworker;
	int32_t n_global_worker = n_procs * n_worker;
	int32_t stage = (global_worker_id + n_global_worker * i_super_cycle) % 3;

	//cout << "Working on stage " << stage << endl;

	double rkold = RKOLD[stage];
	double rknew = RKNEW[stage];

	//cout << "rkold " << rkold << " rknew " << rknew << endl;

	
	
	// Sort out parts
	double* p_c = p_in[0];
	double* p_l = p_in[1];
	double* p_r = p_in[2];
	double* p_ll = p_in[3];
	double* p_rr = p_in[4];

	// Sort out parts
	double* p_c_out = p_out[0];

	// Symmetry BC u0 = -u0
	int sy_bc_ll = 1;
	int sy_bc_l  = 1;
	int sy_bc_r  = 1;
	int sy_bc_rr = 1;

	if (i_part == 0) {
		p_ll = p_rr;
		p_l = p_r;
		sy_bc_ll = -1;
		sy_bc_l  = -1;
	} else if(i_part == 1) {
		p_ll = p_c;
		sy_bc_ll = -1;
	} else if (i_part == my_n_part - 2) {
		p_rr = p_c;
		sy_bc_rr = -1;
	} else if (i_part == my_n_part - 1) {
		p_rr = p_ll;
		p_r = p_l;
		sy_bc_rr = -1;
		sy_bc_r  = -1;
	}

	// Offsets in pages for each field
	size_t offset_rho   = block_header_size + 0*block_ncc;
	size_t offset_rhou0 = block_header_size + 1*block_ncc;
	size_t offset_rhou1 = block_header_size + 2*block_ncc;
	size_t offset_rhou2 = block_header_size + 3*block_ncc;
	size_t offset_rhoE = block_header_size + 4*block_ncc;
	size_t offset_rho_old   = block_header_size + 5*block_ncc;
	size_t offset_rhou0_old = block_header_size + 6*block_ncc;
	size_t offset_rhou1_old = block_header_size + 7*block_ncc;
	size_t offset_rhou2_old = block_header_size + 8*block_ncc;
	size_t offset_rhoE_old = block_header_size + 9*block_ncc;
	size_t offset_tmp0 = block_header_size + 10*block_ncc;
	size_t offset_tmp1 = block_header_size + 11*block_ncc;
	size_t offset_tmp2 = block_header_size + 12*block_ncc;
	size_t offset_tmp3 = block_header_size + 13*block_ncc;
	size_t offset_tmp4 = block_header_size + 14*block_ncc;



	//cout << "Blocksize: " << blockSize << ", threads_per_block: " << threads_per_block << endl;

	threads_per_block = BLOCKSIZE_Z * BLOCKSIZE_Y;
	int32_t gridSize = (blockSize + threads_per_block - 1) / threads_per_block;

	//cout << "Slice Size: " << blockSize << ", gridSize: " << gridSize << ", started Threads: " << gridSize * threads_per_block << endl;


	if (stage == 0) {
		dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rho], &p_c[offset_rho_old]);
		dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou0], &p_c[offset_rhou0_old]);
		dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou1], &p_c[offset_rhou1_old]);
		dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou2], &p_c[offset_rhou2_old]);
		dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhoE], &p_c[offset_rhoE_old]);
	}

	// Constituent relations
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou0], &p_c[offset_rho], (double*) d_u0_c);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou0], &p_l[offset_rho], (double*) d_u0_l);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_r[offset_rhou0], &p_r[offset_rho], (double*) d_u0_r);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_ll[offset_rhou0], &p_ll[offset_rho], (double*) d_u0_ll);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_rr[offset_rhou0], &p_rr[offset_rho], (double*) d_u0_rr);

	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou1], &p_c[offset_rho], (double*) d_u1_c);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou1], &p_l[offset_rho], (double*) d_u1_l);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_r[offset_rhou1], &p_r[offset_rho], (double*) d_u1_r);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_ll[offset_rhou1], &p_ll[offset_rho], (double*) d_u1_ll);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_rr[offset_rhou1], &p_rr[offset_rho], (double*) d_u1_rr);

	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou2], &p_c[offset_rho], (double*) d_u2_c);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou2], &p_l[offset_rho], (double*) d_u2_l);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_r[offset_rhou2], &p_r[offset_rho], (double*) d_u2_r);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_ll[offset_rhou2], &p_ll[offset_rho], (double*) d_u2_ll);
	dns_fdivg <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_rr[offset_rhou2], &p_rr[offset_rho], (double*) d_u2_rr);

	dns_p <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rho], (double*) d_u0_c, (double*) d_u1_c, (double*) d_u2_c, &p_c[offset_rhoE], (double*) d_p_c);
	dns_p <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rho], (double*) d_u0_l, (double*) d_u1_l, (double*) d_u2_l, &p_l[offset_rhoE], (double*) d_p_l);
	dns_p <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_r[offset_rho], (double*) d_u0_r, (double*) d_u1_r, (double*) d_u2_r, &p_r[offset_rhoE], (double*) d_p_r);
	dns_p <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_ll[offset_rho], (double*) d_u0_ll, (double*) d_u1_ll, (double*) d_u2_ll, &p_ll[offset_rhoE], (double*) d_p_ll);
	dns_p <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_rr[offset_rho], (double*) d_u0_rr, (double*) d_u1_rr, (double*) d_u2_rr, &p_rr[offset_rhoE], (double*) d_p_rr);

	dns_T <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rho], (double*) d_p_c, (double*) d_T_c);
	dns_T <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rho], (double*) d_p_l, (double*) d_T_l);
	dns_T <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_r[offset_rho], (double*) d_p_r, (double*) d_T_r);
	dns_T <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_ll[offset_rho], (double*) d_p_ll, (double*) d_T_ll);
	dns_T <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_rr[offset_rho], (double*) d_p_rr, (double*) d_T_rr);


	// du0dx, du0dy, du0dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_du0dx, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u0_c, (double*) d_du0dy, (double*) d_du0dz);

	// du1dx, du1dy, du1dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u1_l, (double*) d_u1_r, (double*) d_u1_ll, (double*) d_u1_rr, (double*) d_du1dx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u1_c, (double*) d_du1dy, (double*) d_du1dz);

	// du2dx, du2dy, du2dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u2_l, (double*) d_u2_r, (double*) d_u2_ll, (double*) d_u2_rr, (double*) d_du2dx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u2_c, (double*) d_du2dy, (double*) d_du2dz);

	// drhodx, drhody, drhodz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rho], &p_r[offset_rho], &p_ll[offset_rho], &p_rr[offset_rho], (double*) d_drhodx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rho], (double*) d_drhody, (double*) d_drhodz);

	// drhou0dx, drhou0dy, drhou0dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou0], &p_r[offset_rhou0], &p_ll[offset_rhou0], &p_rr[offset_rhou0], (double*) d_drhou0dx, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou0], (double*) d_drhou0dy, (double*) d_drhou0dz);
	
	// drhou1dx, drhou1dy, drhou1dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou1], &p_r[offset_rhou1], &p_ll[offset_rhou1], &p_rr[offset_rhou1], (double*) d_drhou1dx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou1], (double*) d_drhou1dy, (double*) d_drhou1dz);

	// drhou2dx, drhou2dy, drhou2dz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou2], &p_r[offset_rhou2], &p_ll[offset_rhou2], &p_rr[offset_rhou2], (double*) d_drhou2dx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou2], (double*) d_drhou2dy, (double*) d_drhou2dz);

	// dpdx, dpdy, dpdz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_p_l, (double*) d_p_r, (double*) d_p_ll, (double*) d_p_rr, (double*) d_dpdx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_p_c, (double*) d_dpdy, (double*) d_dpdz);

	// drhou0u0dx, drhou0u1dy, drhou0u2dz
	dns_dfgdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou0], &p_r[offset_rhou0], &p_ll[offset_rhou0], &p_rr[offset_rhou0],
																												(double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_drhou0u0dx, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfgdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou0], (double*) d_u1_c, (double*) d_drhou0u1dy);
	dns_dfgdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou0], (double*) d_u2_c, (double*) d_drhou0u2dz);

	// drhou1u0dx, drhou1u1dy, drhou1u2dz
	dns_dfgdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou1], &p_r[offset_rhou1], &p_ll[offset_rhou1], &p_rr[offset_rhou1],
																												(double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_drhou1u0dx, 1, 1, 1, 1, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfgdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou1], (double*) d_u1_c, (double*) d_drhou1u1dy);
	dns_dfgdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou1], (double*) d_u2_c, (double*) d_drhou1u2dz);

	// drhou2u0dx, drhou2u1dy, drhou2u2dz
	dns_dfgdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhou2], &p_r[offset_rhou2], &p_ll[offset_rhou2], &p_rr[offset_rhou2],
																												(double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_drhou2u0dx, 1, 1, 1, 1, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfgdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou2], (double*) d_u1_c, (double*) d_drhou2u1dy);
	dns_dfgdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhou2], (double*) d_u2_c, (double*) d_drhou2u2dz);

	// du0d2x, du0d2y, du0d2z
	dns_dfd2x <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u0_c, (double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_du0d2x, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfd2yz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u0_c, (double*) d_du0d2y, (double*) d_du0d2z);

	// du1d2x, du1d2y, du1d2z
	dns_dfd2x <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u1_c, (double*) d_u1_l, (double*) d_u1_r, (double*) d_u1_ll, (double*) d_u1_rr, (double*) d_du1d2x);
	dns_dfd2yz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u1_c, (double*) d_du1d2y, (double*) d_du1d2z);

	// du2d2x, du2d2y, du2d2z
	dns_dfd2x <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u2_c, (double*) d_u2_l, (double*) d_u2_r, (double*) d_u2_ll, (double*) d_u2_rr, (double*) d_du2d2x);
	dns_dfd2yz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_u2_c, (double*) d_du2d2y, (double*) d_du2d2z);

	//14 *3		
	// du0dxdy, du0dxdz
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_du0dx, (double*) d_du0dxdy, (double*) d_du0dxdz);

	// ddns_dfgdz
	dns_dfdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_du1dx, (double*) d_du1dxdy);
	dns_dfdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_du1dy, (double*) d_du1dydz);

	// du2dxdz, du2dydz
	dns_dfdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_du2dx, (double*) d_du2dxdz);
	dns_dfdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_du2dy, (double*) d_du2dydz);

	// 3 * 2
	// dpu0dx, dpu1dy, dpu2dz
	dns_dfgdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_p_l, (double*) d_p_r, (double*) d_p_ll, (double*) d_p_rr,
																												(double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_dpu0dx, 1, 1, 1, 1, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfgdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_p_c, (double*) d_u1_c, (double*) d_dpu1dy);
	dns_dfgdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_p_c, (double*) d_u2_c, (double*) d_dpu2dz);

	// dTd2x, dTd2y, dTd2z
	dns_dfd2x <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_T_c, (double*) d_T_l, (double*) d_T_r, (double*) d_T_ll, (double*) d_T_rr, (double*) d_dTd2x);
	dns_dfd2yz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_T_c, (double*) d_dTd2y, (double*) d_dTd2z);

	// drhoEdx, drhoEdy, drhoEdz
	dns_dfdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhoE], &p_r[offset_rhoE], &p_ll[offset_rhoE], &p_rr[offset_rhoE], (double*) d_drhoEdx);
	dns_dfdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhoE], (double*) d_drhoEdy, (double*) d_drhoEdz);

	// drhoEu0dx, drhoEu1dy, drhoEu2dz
	dns_dfgdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_l[offset_rhoE], &p_r[offset_rhoE], &p_ll[offset_rhoE], &p_rr[offset_rhoE],
																												(double*) d_u0_l, (double*) d_u0_r, (double*) d_u0_ll, (double*) d_u0_rr, (double*) d_drhoEu0dx, 1, 1, 1, 1, sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr);
	dns_dfgdy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhoE], (double*) d_u1_c, (double*) d_drhoEu1dy);
	dns_dfgdz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, &p_c[offset_rhoE], (double*) d_u2_c, (double*) d_drhoEu2dz);

	// 4 * 3

	dns_Res <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
						(double*) &p_c[offset_rho],
						(double*) d_u0_c, (double*) d_u1_c, (double*) d_u2_c,
						&p_c[offset_rhou0], &p_c[offset_rhou1], &p_c[offset_rhou2], &p_c[offset_rhoE],
						(double*) d_p_c, (double*) d_T_c,
						(double*) d_du0dx, (double*) d_du0dy, (double*) d_du0dz,
						(double*) d_du1dx, (double*) d_du1dy, (double*) d_du1dz,
						(double*) d_du2dx, (double*) d_du2dy, (double*) d_du2dz,
						(double*) d_drhodx, (double*) d_drhody, (double*) d_drhodz,
						(double*) d_drhou0dx, (double*) d_drhou0dy, (double*) d_drhou0dz,
						(double*) d_drhou1dx, (double*) d_drhou1dy, (double*) d_drhou1dz,
						(double*) d_drhou2dx, (double*) d_drhou2dy, (double*) d_drhou2dz,
						(double*) d_dpdx, (double*) d_dpdy, (double*) d_dpdz,
						(double*) d_dpu0dx, (double*) d_dpu1dy, (double*) d_dpu2dz,
						(double*) d_drhou0u0dx, (double*) d_drhou0u1dy, (double*) d_drhou0u2dz,
						(double*) d_drhou1u0dx, (double*) d_drhou1u1dy, (double*) d_drhou1u2dz,
						(double*) d_drhou2u0dx, (double*) d_drhou2u1dy, (double*) d_drhou2u2dz,
						(double*) d_du0d2x, (double*) d_du0d2y, (double*) d_du0d2z,
						(double*) d_du1d2x, (double*) d_du1d2y, (double*) d_du1d2z,
						(double*) d_du2d2x, (double*) d_du2d2y, (double*) d_du2d2z,
						(double*) d_du0dxdy, (double*) d_du0dxdz,
 						(double*) d_du1dxdy, (double*) d_du1dydz,
 						(double*) d_du2dxdz, (double*) d_du2dydz,
						(double*) d_dTd2x, (double*) d_dTd2y, (double*) d_dTd2z,
						(double*) d_drhoEdx, (double*) d_drhoEdy, (double*) d_drhoEdz,
						(double*) d_drhoEu0dx, (double*) d_drhoEu1dy, (double*) d_drhoEu2dz,
						(double*) d_Res_rho,
						(double*) d_Res_rhou0,
						(double*) d_Res_rhou1,
						(double*) d_Res_rhou2,
						(double*) d_Res_rhoE);

		dns_RKsubStage <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
					&p_c_out[offset_rho], 
					&p_c_out[offset_rhou0], &p_c_out[offset_rhou1], &p_c_out[offset_rhou2],
					&p_c_out[offset_rhoE],
					&p_c[offset_rho_old],
					&p_c[offset_rhou0_old], &p_c[offset_rhou1_old], &p_c[offset_rhou2_old],
					&p_c[offset_rhoE_old],
					(double*) d_Res_rho,
					(double*) d_Res_rhou0, (double*) d_Res_rhou1, (double*) d_Res_rhou2, 
					(double*) d_Res_rhoE,
					rknew);
	
	dns_RKtmpAdvance <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
					&p_c_out[offset_rho_old],
					&p_c_out[offset_rhou0_old], &p_c_out[offset_rhou1_old], &p_c_out[offset_rhou2_old],
					&p_c_out[offset_rhoE_old],
					&p_c[offset_rho_old],
					&p_c[offset_rhou0_old], &p_c[offset_rhou1_old], &p_c[offset_rhou2_old],
					&p_c[offset_rhoE_old],
					(double*) d_Res_rho,
					(double*) d_Res_rhou0, (double*) d_Res_rhou1, (double*) d_Res_rhou2, 
					(double*) d_Res_rhoE,
					rkold);

	//dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_Res_rho, &p_c[offset_tmp0]);
	//dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_Res_rhou0, &p_c[offset_tmp1]);
	//dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_Res_rhou1, &p_c[offset_tmp2]);
	//dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_Res_rhou2, &p_c[offset_tmp3]);
	//dns_copy <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, (double*) d_Res_rhoE, &p_c[offset_tmp4]);

	// Copy Header
	dns_init <<<gridSize,threads_per_block,0,*stream>>>((const double*)p_c, (double*)p_c_out);
	//cudaMemcpy((void*)p_c_out,(const void*)p_c,block_header_size * sizeof(double),cudaMemcpyDeviceToDevice); cudaCheckError(__LINE__,__FILE__);
	// 4

}


__global__ void prepare_visual_rectilinear(double * __restrict__ p_in, double * __restrict__ p_out) {

	int32_t global_id = blockIdx.x*blockDim.x+threadIdx.x;
	// int32_t n_threads = blockDim.x*gridDim.x;

	// int32_t * p_in_i32 = (int32_t *)p_in;
	int64_t * p_in_i64 = (int64_t*)p_in;
	double * p_in_d = (double*)p_in;


	int64_t i_part=p_in_i64[0];

	// if (global_id==0) {
	// 	printf("part:%i\n",i_part);
	// }

	double * p_out_double=(double*)p_out;


	// if (global_id==0) {
	// 	p_out_i32[0]=n_mol;
	// 	p_out_i32[1]=i_part;
	// }
	if (global_id<block_ncc) {
		int32_t i_cell=global_id;
		int32_t i_x=i_part;
		int32_t i_y=i_cell/my_n_part;
		int32_t i_z=i_cell-i_y*my_n_part;

		//printf("i_part_%i_%i_%i_%i_\n",i_x,i_y,i_z,i_part);

		for (int32_t i_field=0;i_field<block_n_fields;i_field++) {
			double dtmp=p_in_d[block_header_size+i_field*block_ncc+i_cell];

			/*
			if (i_field == 0) {
			if (i_cell < 2) {
				printf(" \n i_cell = %d, dtmp: %lf", i_cell, dtmp);
			}
			if (i_cell == 0) {
				printf("\nI write my part index to %d", i_field*my_n_part*block_ncc+i_z*block_ncc+i_y*my_n_part+i_x);
				printf("\nCalculated from: i_field %d, my_n_part %d, block_ncc %d, i_z %d, i_y %d, i_x %d\n", i_field, my_n_part, block_ncc, i_z, i_y, i_x);
			}
			}
			*/
			

			
			//double dtmp = 5.0;

			p_out_double[i_field*my_n_part*block_ncc+i_z*block_ncc+i_y*my_n_part+i_x]=dtmp;
		}
	}

}

void DS::write_vtr (double * p_data, int32_t i_part, int32_t i_cycle) {
	string FileName;
	FileName.append("/direc/visual_");
	FileName+=to_string(my_n_part);
	FileName.append("_");
	FileName+=to_string(i_cycle);
	// FileName.append("/visual_");
	// FileName+=to_string(i_part);
	FileName.append(".vtr");

	/*
	printf("\np_data: ");
	
	for (int i = 0; i < 27; ++i) {
		if (i % 3 == 0) printf("\n");
		if (i % 9 == 0) printf("\n");
		printf("%f, ", p_data[i]);
		
	}
	printf("\n");
	*/
	
	

	ofstream ofs;
	ofs.open(FileName, ios::out | ios::binary);
	if (ofs) {
		int64_t append_offset=0;
		ofs << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
		ofs << "<RectilinearGrid WholeExtent=\"" << "0 " << my_n_part-1 << " 0 " << my_n_part-1 << " 0 " << my_n_part-1 << "\">" << endl;
		ofs << "<Piece Extent=\"" << "0 " << my_n_part-1 << " 0 " << my_n_part-1 << " 0 " << my_n_part-1 << "\">" << endl;

		ofs << "<PointData Scalars=\"\" Name=\"a\">" << endl;
		ofs << "<DataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhou0\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhou1\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhou2\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhoE\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rho_old\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhou0_old\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhou1_old\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
		
		ofs << "<DataArray type=\"Float64\" Name=\"rhou2_old\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"rhoE_old\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);

		// ============================ TMP Output ============================
		//ofs << "<DataArray type=\"Float64\" Name=\"tmp0\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		//ofs << append_offset;
		//ofs << "\">";
		//ofs << "</DataArray>" << endl;
		//append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
//
		//ofs << "<DataArray type=\"Float64\" Name=\"tmp1\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		//ofs << append_offset;
		//ofs << "\">";
		//ofs << "</DataArray>" << endl;
		//append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
//
		//ofs << "<DataArray type=\"Float64\" Name=\"tmp2\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		//ofs << append_offset;
		//ofs << "\">";
		//ofs << "</DataArray>" << endl;
		//append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
//
		//ofs << "<DataArray type=\"Float64\" Name=\"tmp3\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		//ofs << append_offset;
		//ofs << "\">";
		//ofs << "</DataArray>" << endl;
		//append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
//
		//ofs << "<DataArray type=\"Float64\" Name=\"tmp4\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		//ofs << append_offset;
		//ofs << "\">";
		//ofs << "</DataArray>" << endl;
		//append_offset+=(my_n_part*block_ncc)*sizeof(double)+sizeof(int64_t);
//
		// ============================ TMP Output ============================

		ofs << "</PointData>" << endl;

		ofs << "<Coordinates>" << endl;
		ofs << "<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part+1)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		// ofs << "\" RangeMin=\"0\" RangeMax=\"1.0\">" << endl;
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part+1)*sizeof(double)+sizeof(int64_t);

		ofs << "<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"1\" format=\"appended\" offset=\"";
		ofs << append_offset;
		ofs << "\">";
		// ofs << "\" RangeMin=\"0\" RangeMax=\"1.0\">" << endl;
		ofs << "</DataArray>" << endl;
		append_offset+=(my_n_part+1)*sizeof(double)+sizeof(int64_t);

		ofs << "</Coordinates>" << endl;

		// ofs << "\" NumberOfCells=\"0\">" << endl;
		// ofs << "<PointData Scalars=\"species\">" << endl;
		// ofs << "<DataArray type=\"Float32\" Name=\"species\" format=\"appended\" offset=\"0\" RangeMin=\"0\" RangeMax=\"6\">" << endl;
		// ofs << "</DataArray>" << endl;
		// ofs << "</PointData>" << endl;
		// ofs << "<Points>" << endl;
		// ofs << "<DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"";
		// ofs << n_mol*sizeof(double)+8;
		// ofs << "\" RangeMin=\"0\" RangeMax=\"1.0\">" << endl;
		// ofs << "</DataArray>" << endl;
		// ofs << "</Points>" << endl;
		// ofs << "<Cells>" << endl;
		// ofs << "<DataArray type=\"Int32\" Name=\"connectivity\"></DataArray>" << endl;
		// ofs << "<DataArray type=\"Int32\" Name=\"offsets\"></DataArray>" << endl;
		// ofs << "<DataArray type=\"UInt8\" Name=\"types\"></DataArray>" << endl;
		// ofs << "</Cells>" << endl;
		ofs << "</Piece>" << endl;
		ofs << "</RectilinearGrid>" << endl;
		ofs << "<AppendedData encoding=\"raw\">" << endl;
		ofs << "_";	// mark start of appended data
		ofs.close();
	}

	// write appended data
	int64_t size_append=0;

	// cell data
	for (int32_t i_field=0;i_field<block_n_fields;i_field++) {
		size_append=(my_n_part*block_ncc)*sizeof(double);
		MemToFile(&size_append,sizeof(int64_t),(char*)FileName.c_str(),0);
		MemToFile((int64_t*)&p_data[i_field*my_n_part*block_ncc],size_append,(char*)FileName.c_str(),0);
	}

	// coordinates - same for x,y,z
	double * x_coordinates=new double [my_n_part+1];
	for (int i=0;i<my_n_part+1;i++) {
		x_coordinates[i]=i;
	}
	size_append=(my_n_part+1)*sizeof(double);
	MemToFile(&size_append,sizeof(int64_t),(char*)FileName.c_str(),0);
	MemToFile((int64_t*)x_coordinates,size_append,(char*)FileName.c_str(),0);

	size_append=(my_n_part+1)*sizeof(double);
	MemToFile(&size_append,sizeof(int64_t),(char*)FileName.c_str(),0);
	MemToFile((int64_t*)x_coordinates,size_append,(char*)FileName.c_str(),0);

	size_append=(my_n_part+1)*sizeof(double);
	MemToFile(&size_append,sizeof(int64_t),(char*)FileName.c_str(),0);
	MemToFile((int64_t*)x_coordinates,size_append,(char*)FileName.c_str(),0);
	delete [] x_coordinates;

	// write closing tags
	ofs.open(FileName, ios::out | ios::binary | ios_base::app);
	if (ofs) {
		ofs << "</AppendedData>" << endl;
		ofs << "</VTKFile>" << endl;
		ofs.close();
	}
}

void DS::caller_output_vtk_rectilinear (double * p_in, double * p_out, cudaStream_t * stream, int32_t threads_per_block, int32_t blockSize, int32_t myID, int32_t i_cycle, int32_t i_part) {

	int32_t n_blocks=block_ncc/threads_per_block;
	n_blocks++;

	prepare_visual_rectilinear <<<n_blocks,threads_per_block,0,*stream>>> (p_in,p_out);
	// int32_t * p_my_vis_i32=(int32_t*)p_my_vis;
	// float * p_my_vis_float=(float*)p_my_vis;
	if (i_part==(my_n_part-1)) {
		// last part
		double * p_my_vis_double=new double[block_n_fields*my_n_part*block_ncc];

		cudaDeviceSynchronize();        cudaCheckError(__LINE__,__FILE__);

		size_t copy_size=1;
		copy_size*=block_n_fields;
		copy_size*=my_n_part;
		copy_size*=block_ncc;
		copy_size*=sizeof(double);
		// cout << copy_size << endl;
		cudaError_t cer=cudaMemcpy((void*)p_my_vis_double,(const void*)p_out,copy_size,cudaMemcpyDeviceToHost); //cudaCheckError(__LINE__,__FILE__);
		cout << cer << endl;
		// for (int i=0;i<block_n_fields*my_n_part*block_ncc;i++) cout << p_my_vis_float[i] << endl;

		// string new_dir;
		// new_dir.append("visual/visual_");
		// new_dir+=to_string(i_cycle);

		// boost::filesystem::create_directory(new_dir.c_str());
		write_vtr(p_my_vis_double,0,i_cycle);
		delete [] p_my_vis_double;
	}
}
