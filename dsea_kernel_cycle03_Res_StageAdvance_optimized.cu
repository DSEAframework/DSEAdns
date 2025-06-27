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



__device__ __forceinline__ double dns_pDer1(double v_ll, double v_l, double v_r, double v_rr, double DK) {
	return 1./DK * (1./12. * v_ll - 2./3. * v_l + 2./3. * v_r - 1./12. * v_rr);
}

__device__ __forceinline__ double dns_pDer2(double v_ll, double v_l, double v_c, double v_r, double v_rr, double DK) {
	return 1./(DK*DK) * (-1./12. * v_ll + 4./3. * v_l - 5./2. * v_c + 4./3. * v_r - 1./12. * v_rr);
}

__device__ __forceinline__ double calp(double irhoE, double irho, double irhou0, double irhou1, double irhou2) {
	//return ((GAMA - 1) * (rhoE - 0.5 * rhou0 * rhou0 / rho - 0.5 * rhou1 * rhou1 / rho - 0.5 rhou2 * rhou2 / rho));
	double tmp = (irhoE - 0.5 * irhou0 * irhou0 / irho - 0.5 * irhou1 * irhou1 / irho - 0.5 * irhou2 * irhou2 / irho);
	return (GAMA - 1) * tmp;
}

__device__ __forceinline__ double calT(double ip, double irho) {
	return MINF * MINF * GAMA * ip / irho;
}

__device__ __forceinline__ double calp_ui(double irhoE, double irho, double u0, double u1, double u2) {
	double tmp = (irhoE - 0.5 * irho * u0 * u0 - 0.5 * irho * u1 * u1 - 0.5 * irho * u2 * u2);
	return (GAMA - 1) * tmp;
}

__device__ __forceinline__ double calT_ui(double ip, double iinvrho) {
	return MINF * MINF * GAMA * ip * iinvrho;
}



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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void dns_du0dxyz(int32_t i_worker, int32_t order_in, int32_t order_out, int32_t problemsize,
	int32_t block_size_z, int32_t block_size_y, int32_t warp_size_z, int32_t warp_size_y,
	/*order 0*/ double * __restrict__ irho_c,
	/*order 1*/ double * __restrict__ irho_l, double * __restrict__ irho_r,
	/*order 2*/ double * __restrict__ irho_ll, double * __restrict__ irho_rr,
	/*order 0*/ double * __restrict__ irhou0_c,
	/*order 1*/ double * __restrict__ irhou0_l, double * __restrict__ irhou0_r,
	/*order 2*/ double * __restrict__ irhou0_ll, double * __restrict__ irhou0_rr,
	int sy_bc_ll, int sy_bc_l, int sy_bc_r, int sy_bc_rr,
	/*order 0*/ double * __restrict__ odu0dx,
	/*order 0*/ double * __restrict__ tmp_du0d2xi,
	/*order 0*/ double * __restrict__ tmp_du1d2xi,
	/*order 0*/ double * __restrict__ tmp_du2d2xi) {
	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t gidx;
	// Idx in Block
	int32_t col_in_block, row_in_block;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &col_in_block, &row_in_block);
	gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (gidx<block_ncc) {
		//dfdx[gidx] = tidx;

		int32_t Y, Z;
		int32_t dy_ll, dy_l, dy_r, dy_rr;
		int32_t dz_ll, dz_l, dz_r, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
		IDX((NY+Y-1)%NY, Z, dy_l, NZ);
		IDX((NY+Y+1)%NY, Z, dy_r, NZ);
		IDX((NY+Y+2)%NY, Z, dy_rr, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, dz_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, dz_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);

		double tmp0 = 0;


		double rhou0_dy_ll, rhou0_dy_l, rhou0_dy_c, rhou0_dy_r, rhou0_dy_rr;
		rhou0_dy_ll = irhou0_c[dy_ll];
		rhou0_dy_l = irhou0_c[dy_l];
		rhou0_dy_c = irhou0_c[gidx];
		rhou0_dy_r = irhou0_c[dy_r];
		rhou0_dy_rr = irhou0_c[dy_rr];

		double rho_dy_ll, rho_dy_l, rho_dy_c, rho_dy_r, rho_dy_rr;
		rho_dy_ll = irho_c[dy_ll];
		rho_dy_l = irho_c[dy_l];
		rho_dy_c = irho_c[gidx];
		rho_dy_r = irho_c[dy_r];
		rho_dy_rr = irho_c[dy_rr];

		double u0_dy_ll, u0_dy_l, u0_dy_c, u0_dy_r, u0_dy_rr;
		u0_dy_ll = rhou0_dy_ll / rho_dy_ll;
		u0_dy_l = rhou0_dy_l / rho_dy_l;
		u0_dy_c = rhou0_dy_c / rho_dy_c;
		u0_dy_r = rhou0_dy_r / rho_dy_r;
		u0_dy_rr = rhou0_dy_rr / rho_dy_rr;

		tmp0 += dns_pDer2(u0_dy_ll, u0_dy_l, u0_dy_c, u0_dy_r, u0_dy_rr, DY);


		double rhou0_dz_ll, rhou0_dz_l, rhou0_dz_c, rhou0_dz_r, rhou0_dz_rr;
		rhou0_dz_ll = irhou0_c[dz_ll];
		rhou0_dz_l = irhou0_c[dz_l];
		rhou0_dz_c = irhou0_c[gidx];
		rhou0_dz_r = irhou0_c[dz_r];
		rhou0_dz_rr = irhou0_c[dz_rr];

		double rho_dz_ll, rho_dz_l, rho_dz_c, rho_dz_r, rho_dz_rr;
		rho_dz_ll = irho_c[dz_ll];
		rho_dz_l = irho_c[dz_l];
		rho_dz_c = irho_c[gidx];
		rho_dz_r = irho_c[dz_r];
		rho_dz_rr = irho_c[dz_rr];

		double u0_dz_ll, u0_dz_l, u0_dz_c, u0_dz_r, u0_dz_rr;
		u0_dz_ll = rhou0_dz_ll / rho_dz_ll;
		u0_dz_l = rhou0_dz_l / rho_dz_l;
		u0_dz_c = rhou0_dz_c / rho_dz_c;
		u0_dz_r = rhou0_dz_r / rho_dz_r;
		u0_dz_rr = rhou0_dz_rr / rho_dz_rr;

		tmp0 += dns_pDer2(u0_dz_ll, u0_dz_l, u0_dz_c, u0_dz_r, u0_dz_rr, DZ);


		double rhou0_dx_ll, rhou0_dx_l, rhou0_dx_c, rhou0_dx_r, rhou0_dx_rr;
		rhou0_dx_ll = irhou0_ll[gidx];
		rhou0_dx_l = irhou0_l[gidx];
		rhou0_dx_c = irhou0_c[gidx];
		rhou0_dx_r = irhou0_r[gidx];
		rhou0_dx_rr = irhou0_rr[gidx];

		double rho_dx_ll, rho_dx_l, rho_dx_c, rho_dx_r, rho_dx_rr;
		rho_dx_ll = irho_ll[gidx];
		rho_dx_l = irho_l[gidx];
		rho_dx_c = irho_c[gidx];
		rho_dx_r = irho_r[gidx];
		rho_dx_rr = irho_rr[gidx];

		double u0_dx_ll, u0_dx_l, u0_dx_c, u0_dx_r, u0_dx_rr;
		u0_dx_ll = rhou0_dx_ll / rho_dx_ll;
		u0_dx_l = rhou0_dx_l / rho_dx_l;
		u0_dx_c = rhou0_dx_c / rho_dx_c;
		u0_dx_r = rhou0_dx_r / rho_dx_r;
		u0_dx_rr = rhou0_dx_rr / rho_dx_rr;

		odu0dx[gidx] = dns_pDer1(sy_bc_ll *  u0_dx_ll, sy_bc_l * u0_dx_l, sy_bc_r * u0_dx_r, sy_bc_rr * u0_dx_rr, DX);
		tmp0 += 4./3. * dns_pDer2(sy_bc_ll * u0_dx_ll, sy_bc_l * u0_dx_l, u0_dx_c, sy_bc_r * u0_dx_r, sy_bc_rr * u0_dx_rr, DX);

		tmp_du0d2xi[gidx] = tmp0;

		double u0_dy_ll_dx_ll = irhou0_ll[dy_ll] / irho_ll[dy_ll];
		double u0_dy_l_dx_ll = irhou0_ll[dy_l] / irho_ll[dy_l];
		double u0_dy_r_dx_ll = irhou0_ll[dy_r] / irho_ll[dy_r];
		double u0_dy_rr_dx_ll = irhou0_ll[dy_rr] / irho_ll[dy_rr];

		double u0_dz_ll_dx_ll = irhou0_ll[dz_ll] / irho_ll[dz_ll];
		double u0_dz_l_dx_ll = irhou0_ll[dz_l] / irho_ll[dz_l];
		double u0_dz_r_dx_ll = irhou0_ll[dz_r] / irho_ll[dz_r];
		double u0_dz_rr_dx_ll = irhou0_ll[dz_rr] / irho_ll[dz_rr];

		double du0dy_dx_ll = dns_pDer1(u0_dy_ll_dx_ll, u0_dy_l_dx_ll, u0_dy_r_dx_ll, u0_dy_rr_dx_ll, DY);
		double du0dz_dx_ll = dns_pDer1(u0_dz_ll_dx_ll, u0_dz_l_dx_ll, u0_dz_r_dx_ll, u0_dz_rr_dx_ll, DZ);

		double u0_dy_ll_dx_l = irhou0_l[dy_ll] / irho_l[dy_ll];
		double u0_dy_l_dx_l = irhou0_l[dy_l] / irho_l[dy_l];
		double u0_dy_r_dx_l = irhou0_l[dy_r] / irho_l[dy_r];
		double u0_dy_rr_dx_l = irhou0_l[dy_rr] / irho_l[dy_rr];

		double u0_dz_ll_dx_l = irhou0_l[dz_ll] / irho_l[dz_ll];
		double u0_dz_l_dx_l = irhou0_l[dz_l] / irho_l[dz_l];
		double u0_dz_r_dx_l = irhou0_l[dz_r] / irho_l[dz_r];
		double u0_dz_rr_dx_l = irhou0_l[dz_rr] / irho_l[dz_rr];

		double du0dy_dx_l = dns_pDer1(u0_dy_ll_dx_l, u0_dy_l_dx_l, u0_dy_r_dx_l, u0_dy_rr_dx_l, DY);
		double du0dz_dx_l = dns_pDer1(u0_dz_ll_dx_l, u0_dz_l_dx_l, u0_dz_r_dx_l, u0_dz_rr_dx_l, DZ);

		double u0_dy_ll_dx_r = irhou0_r[dy_ll] / irho_r[dy_ll];
		double u0_dy_l_dx_r = irhou0_r[dy_l] / irho_r[dy_l];
		double u0_dy_r_dx_r = irhou0_r[dy_r] / irho_r[dy_r];
		double u0_dy_rr_dx_r = irhou0_r[dy_rr] / irho_r[dy_rr];

		double u0_dz_ll_dx_r = irhou0_r[dz_ll] / irho_r[dz_ll];
		double u0_dz_l_dx_r = irhou0_r[dz_l] / irho_r[dz_l];
		double u0_dz_r_dx_r = irhou0_r[dz_r] / irho_r[dz_r];
		double u0_dz_rr_dx_r = irhou0_r[dz_rr] / irho_r[dz_rr];

		double du0dy_dx_r = dns_pDer1(u0_dy_ll_dx_r, u0_dy_l_dx_r, u0_dy_r_dx_r, u0_dy_rr_dx_r, DY);
		double du0dz_dx_r = dns_pDer1(u0_dz_ll_dx_r, u0_dz_l_dx_r, u0_dz_r_dx_r, u0_dz_rr_dx_r, DZ);

		double u0_dy_ll_dx_rr = irhou0_rr[dy_ll] / irho_rr[dy_ll];
		double u0_dy_l_dx_rr = irhou0_rr[dy_l] / irho_rr[dy_l];
		double u0_dy_r_dx_rr = irhou0_rr[dy_r] / irho_rr[dy_r];
		double u0_dy_rr_dx_rr = irhou0_rr[dy_rr] / irho_rr[dy_rr];

		double u0_dz_ll_dx_rr = irhou0_rr[dz_ll] / irho_rr[dz_ll];
		double u0_dz_l_dx_rr = irhou0_rr[dz_l] / irho_rr[dz_l];
		double u0_dz_r_dx_rr = irhou0_rr[dz_r] / irho_rr[dz_r];
		double u0_dz_rr_dx_rr = irhou0_rr[dz_rr] / irho_rr[dz_rr];

		double du0dy_dx_rr = dns_pDer1(u0_dy_ll_dx_rr, u0_dy_l_dx_rr, u0_dy_r_dx_rr, u0_dy_rr_dx_rr, DY);
		double du0dz_dx_rr = dns_pDer1(u0_dz_ll_dx_rr, u0_dz_l_dx_rr, u0_dz_r_dx_rr, u0_dz_rr_dx_rr, DZ);

		tmp_du1d2xi[gidx] = 1./3. * dns_pDer1(sy_bc_ll *  du0dy_dx_ll, sy_bc_l * du0dy_dx_l, sy_bc_r * du0dy_dx_r, sy_bc_rr * du0dy_dx_rr, DX);
		tmp_du2d2xi[gidx] = 1./3. * dns_pDer1(sy_bc_ll *  du0dz_dx_ll, sy_bc_l * du0dz_dx_l, sy_bc_r * du0dz_dx_r, sy_bc_rr * du0dz_dx_rr, DX);

	}
}

__global__ void dns_du1dxyz(int32_t i_worker, int32_t order_in, int32_t order_out, int32_t problemsize,
	int32_t block_size_z, int32_t block_size_y, int32_t warp_size_z, int32_t warp_size_y,
	/*order 0*/ double * __restrict__ irho_c,
	/*order 1*/ double * __restrict__ irho_l, double * __restrict__ irho_r,
	/*order 2*/ double * __restrict__ irho_ll, double * __restrict__ irho_rr,
	/*order 0*/ double * __restrict__ irhou1_c,
	/*order 1*/ double * __restrict__ irhou1_l, double * __restrict__ irhou1_r,
	/*order 2*/ double * __restrict__ irhou1_ll, double * __restrict__ irhou1_rr,
	/*order 0*/ double * __restrict__ odu1dx,
  /*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ tmp_du0d2xi,
	/*order 0*/ double * __restrict__ tmp_du1d2xi,
	/*order 0*/ double * __restrict__ tmp_du2d2xi) {
	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t gidx;
	// Idx in Block
	int32_t col_in_block, row_in_block;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &col_in_block, &row_in_block);
	gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (gidx<block_ncc) {
		//dfdx[gidx] = tidx;

		int32_t Y, Z;
		int32_t dy_ll, dy_l, dy_r, dy_rr;
		int32_t dz_ll, dz_l, dz_r, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
		IDX((NY+Y-1)%NY, Z, dy_l, NZ);
		IDX((NY+Y+1)%NY, Z, dy_r, NZ);
		IDX((NY+Y+2)%NY, Z, dy_rr, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, dz_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, dz_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);

		double tmp0 = 0;


		double rhou1_dy_ll, rhou1_dy_l, rhou1_dy_c, rhou1_dy_r, rhou1_dy_rr;
		rhou1_dy_ll = irhou1_c[dy_ll];
		rhou1_dy_l = irhou1_c[dy_l];
		rhou1_dy_c = irhou1_c[gidx];
		rhou1_dy_r = irhou1_c[dy_r];
		rhou1_dy_rr = irhou1_c[dy_rr];

		double rho_dy_ll, rho_dy_l, rho_dy_c, rho_dy_r, rho_dy_rr;
		rho_dy_ll = irho_c[dy_ll];
		rho_dy_l = irho_c[dy_l];
		rho_dy_c = irho_c[gidx];
		rho_dy_r = irho_c[dy_r];
		rho_dy_rr = irho_c[dy_rr];

		double u1_dy_ll, u1_dy_l, u1_dy_c, u1_dy_r, u1_dy_rr;
		u1_dy_ll = rhou1_dy_ll / rho_dy_ll;
		u1_dy_l = rhou1_dy_l / rho_dy_l;
		u1_dy_c = rhou1_dy_c / rho_dy_c;
		u1_dy_r = rhou1_dy_r / rho_dy_r;
		u1_dy_rr = rhou1_dy_rr / rho_dy_rr;

		tmp0 += 4./3. * dns_pDer2(u1_dy_ll, u1_dy_l, u1_dy_c, u1_dy_r, u1_dy_rr, DY);

		double rhou1_dz_ll, rhou1_dz_l, rhou1_dz_c, rhou1_dz_r, rhou1_dz_rr;
		rhou1_dz_ll = irhou1_c[dz_ll];
		rhou1_dz_l = irhou1_c[dz_l];
		rhou1_dz_c = irhou1_c[gidx];
		rhou1_dz_r = irhou1_c[dz_r];
		rhou1_dz_rr = irhou1_c[dz_rr];

		double rho_dz_ll, rho_dz_l, rho_dz_c, rho_dz_r, rho_dz_rr;
		rho_dz_ll = irho_c[dz_ll];
		rho_dz_l = irho_c[dz_l];
		rho_dz_c = irho_c[gidx];
		rho_dz_r = irho_c[dz_r];
		rho_dz_rr = irho_c[dz_rr];

		double u1_dz_ll, u1_dz_l, u1_dz_c, u1_dz_r, u1_dz_rr;
		u1_dz_ll = rhou1_dz_ll / rho_dz_ll;
		u1_dz_l = rhou1_dz_l / rho_dz_l;
		u1_dz_c = rhou1_dz_c / rho_dz_c;
		u1_dz_r = rhou1_dz_r / rho_dz_r;
		u1_dz_rr = rhou1_dz_rr / rho_dz_rr;

		tmp0 += dns_pDer2(u1_dz_ll, u1_dz_l, u1_dz_c, u1_dz_r, u1_dz_rr, DZ);


		double rhou1_dx_ll, rhou1_dx_l, rhou1_dx_c, rhou1_dx_r, rhou1_dx_rr;
		rhou1_dx_ll = irhou1_ll[gidx];
		rhou1_dx_l = irhou1_l[gidx];
		rhou1_dx_c = irhou1_c[gidx];
		rhou1_dx_r = irhou1_r[gidx];
		rhou1_dx_rr = irhou1_rr[gidx];

		double rho_dx_ll, rho_dx_l, rho_dx_c, rho_dx_r, rho_dx_rr;
		rho_dx_ll = irho_ll[gidx];
		rho_dx_l = irho_l[gidx];
		rho_dx_c = irho_c[gidx];
		rho_dx_r = irho_r[gidx];
		rho_dx_rr = irho_rr[gidx];

		double u1_dx_ll, u1_dx_l, u1_dx_c, u1_dx_r, u1_dx_rr;
		u1_dx_ll = rhou1_dx_ll / rho_dx_ll;
		u1_dx_l = rhou1_dx_l / rho_dx_l;
		u1_dx_c = rhou1_dx_c / rho_dx_c;
		u1_dx_r = rhou1_dx_r / rho_dx_r;
		u1_dx_rr = rhou1_dx_rr / rho_dx_rr;

		odu1dx[gidx] = dns_pDer1(u1_dx_ll, u1_dx_l, u1_dx_r, u1_dx_rr, DX);
		tmp0 += dns_pDer2(u1_dx_ll, u1_dx_l, u1_dx_c, u1_dx_r, u1_dx_rr, DX);

		tmp_du1d2xi[gidx] += tmp0;


		double u1_dx_ll_dy_ll = irhou1_ll[dy_ll] / irho_ll[dy_ll];
		double u1_dx_l_dy_ll = irhou1_l[dy_ll] / irho_l[dy_ll];
		double u1_dx_r_dy_ll = irhou1_r[dy_ll] / irho_r[dy_ll];
		double u1_dx_rr_dy_ll = irhou1_rr[dy_ll] / irho_rr[dy_ll];

		double du1dx_dy_ll = dns_pDer1(u1_dx_ll_dy_ll, u1_dx_l_dy_ll, u1_dx_r_dy_ll, u1_dx_rr_dy_ll, DX);

		double u1_dx_ll_dy_l = irhou1_ll[dy_l] / irho_ll[dy_l];
		double u1_dx_l_dy_l = irhou1_l[dy_l] / irho_l[dy_l];
		double u1_dx_r_dy_l = irhou1_r[dy_l] / irho_r[dy_l];
		double u1_dx_rr_dy_l = irhou1_rr[dy_l] / irho_rr[dy_l];

		double du1dx_dy_l = dns_pDer1(u1_dx_ll_dy_l, u1_dx_l_dy_l, u1_dx_r_dy_l, u1_dx_rr_dy_l, DX);

		double u1_dx_ll_dy_r = irhou1_ll[dy_r] / irho_ll[dy_r];
		double u1_dx_l_dy_r = irhou1_l[dy_r] / irho_l[dy_r];
		double u1_dx_r_dy_r = irhou1_r[dy_r] / irho_r[dy_r];
		double u1_dx_rr_dy_r = irhou1_rr[dy_r] / irho_rr[dy_r];

		double du1dx_dy_r = dns_pDer1(u1_dx_ll_dy_r, u1_dx_l_dy_r, u1_dx_r_dy_r, u1_dx_rr_dy_r, DX);

		double u1_dx_ll_dy_rr = irhou1_ll[dy_rr] / irho_ll[dy_rr];
		double u1_dx_l_dy_rr = irhou1_l[dy_rr] / irho_l[dy_rr];
		double u1_dx_r_dy_rr = irhou1_r[dy_rr] / irho_r[dy_rr];
		double u1_dx_rr_dy_rr = irhou1_rr[dy_rr] / irho_rr[dy_rr];

		double du1dx_dy_rr = dns_pDer1(u1_dx_ll_dy_rr, u1_dx_l_dy_rr, u1_dx_r_dy_rr, u1_dx_rr_dy_rr, DX);

		tmp_du0d2xi[gidx] += 1./3. * dns_pDer1(du1dx_dy_ll, du1dx_dy_l, du1dx_dy_r, du1dx_dy_rr, DY);

		// Calculate idx with periodic boundary condition
		int32_t dy_ll_dz_ll, dy_ll_dz_l, dy_ll_dz_r, dy_ll_dz_rr;
		IDX((NY+Y-2)%NY, (NZ+Z-2)%NZ, dy_ll_dz_ll, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z-1)%NZ, dy_ll_dz_l, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z+1)%NZ, dy_ll_dz_r, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z+2)%NZ, dy_ll_dz_rr, NZ);

		double u1_dz_ll_dy_ll = irhou1_c[dy_ll_dz_ll] / irho_c[dy_ll_dz_ll];
		double u1_dz_l_dy_ll = irhou1_c[dy_ll_dz_l] / irho_c[dy_ll_dz_l];
		double u1_dz_r_dy_ll = irhou1_c[dy_ll_dz_r] / irho_c[dy_ll_dz_r];
		double u1_dz_rr_dy_ll = irhou1_c[dy_ll_dz_rr] / irho_c[dy_ll_dz_rr];

		double du1dz_dy_ll = dns_pDer1(u1_dz_ll_dy_ll, u1_dz_l_dy_ll, u1_dz_r_dy_ll, u1_dz_rr_dy_ll, DZ);


		int32_t dy_l_dz_ll, dy_l_dz_l, dy_l_dz_r, dy_l_dz_rr;
		IDX((NY+Y-1)%NY, (NZ+Z-2)%NZ, dy_l_dz_ll, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z-1)%NZ, dy_l_dz_l, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z+1)%NZ, dy_l_dz_r, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z+2)%NZ, dy_l_dz_rr, NZ);

		double u1_dz_ll_dy_l = irhou1_c[dy_l_dz_ll] / irho_c[dy_l_dz_ll];
		double u1_dz_l_dy_l = irhou1_c[dy_l_dz_l] / irho_c[dy_l_dz_l];
		double u1_dz_r_dy_l = irhou1_c[dy_l_dz_r] / irho_c[dy_l_dz_r];
		double u1_dz_rr_dy_l = irhou1_c[dy_l_dz_rr] / irho_c[dy_l_dz_rr];

		double du1dz_dy_l = dns_pDer1(u1_dz_ll_dy_l, u1_dz_l_dy_l, u1_dz_r_dy_l, u1_dz_rr_dy_l, DZ);


		int32_t dy_r_dz_ll, dy_r_dz_l, dy_r_dz_r, dy_r_dz_rr;
		IDX((NY+Y+1)%NY, (NZ+Z-2)%NZ, dy_r_dz_ll, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z-1)%NZ, dy_r_dz_l, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z+1)%NZ, dy_r_dz_r, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z+2)%NZ, dy_r_dz_rr, NZ);

		double u1_dz_ll_dy_r = irhou1_c[dy_r_dz_ll] / irho_c[dy_r_dz_ll];
		double u1_dz_l_dy_r = irhou1_c[dy_r_dz_l] / irho_c[dy_r_dz_l];
		double u1_dz_r_dy_r = irhou1_c[dy_r_dz_r] / irho_c[dy_r_dz_r];
		double u1_dz_rr_dy_r = irhou1_c[dy_r_dz_rr] / irho_c[dy_r_dz_rr];

		double du1dz_dy_r = dns_pDer1(u1_dz_ll_dy_r, u1_dz_l_dy_r, u1_dz_r_dy_r, u1_dz_rr_dy_r, DZ);


		int32_t dy_rr_dz_ll, dy_rr_dz_l, dy_rr_dz_r, dy_rr_dz_rr;
		IDX((NY+Y+2)%NY, (NZ+Z-2)%NZ, dy_rr_dz_ll, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z-1)%NZ, dy_rr_dz_l, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z+1)%NZ, dy_rr_dz_r, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z+2)%NZ, dy_rr_dz_rr, NZ);

		double u1_dz_ll_dy_rr = irhou1_c[dy_rr_dz_ll] / irho_c[dy_rr_dz_ll];
		double u1_dz_l_dy_rr = irhou1_c[dy_rr_dz_l] / irho_c[dy_rr_dz_l];
		double u1_dz_r_dy_rr = irhou1_c[dy_rr_dz_r] / irho_c[dy_rr_dz_r];
		double u1_dz_rr_dy_rr = irhou1_c[dy_rr_dz_rr] / irho_c[dy_rr_dz_rr];

		double du1dz_dy_rr = dns_pDer1(u1_dz_ll_dy_rr, u1_dz_l_dy_rr, u1_dz_r_dy_rr, u1_dz_rr_dy_rr, DZ);

		tmp_du2d2xi[gidx] += 1./3. * dns_pDer1(du1dz_dy_ll,  du1dz_dy_l, du1dz_dy_r, du1dz_dy_rr, DY);
	}
}

__global__ void dns_du2dxyz(int32_t i_worker, int32_t order_in, int32_t order_out, int32_t problemsize,
	int32_t block_size_z, int32_t block_size_y, int32_t warp_size_z, int32_t warp_size_y,
	/*order 0*/ double * __restrict__ irho_c,
	/*order 1*/ double * __restrict__ irho_l, double * __restrict__ irho_r,
	/*order 2*/ double * __restrict__ irho_ll, double * __restrict__ irho_rr,
	/*order 0*/ double * __restrict__ irhou2_c,
	/*order 1*/ double * __restrict__ irhou2_l, double * __restrict__ irhou2_r,
	/*order 2*/ double * __restrict__ irhou2_ll, double * __restrict__ irhou2_rr,
	/*order 0*/ double * __restrict__ odu2dx,
	/*order 0*/ double * __restrict__ tmp_du0d2xi,
	/*order 0*/ double * __restrict__ tmp_du1d2xi,
	/*order 0*/ double * __restrict__ tmp_du2d2xi) {
	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t gidx;
	// Idx in Block
	int32_t col_in_block, row_in_block;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &col_in_block, &row_in_block);
	gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);


	if (gidx<block_ncc) {
		//dfdx[gidx] = tidx;

		int32_t Y, Z;
		int32_t dy_ll, dy_l, dy_r, dy_rr;
		int32_t dz_ll, dz_l, dz_r, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// Calculate idx with periodic boundary condition
		IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
		IDX((NY+Y-1)%NY, Z, dy_l, NZ);
		IDX((NY+Y+1)%NY, Z, dy_r, NZ);
		IDX((NY+Y+2)%NY, Z, dy_rr, NZ);

		// Calculate idx with periodic boundary condition
		IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
		IDX(Y, (NZ+Z-1)%NZ, dz_l, NZ);
		IDX(Y, (NZ+Z+1)%NZ, dz_r, NZ);
		IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);

		double tmp0 = 0;

		double rhou2_dy_ll, rhou2_dy_l, rhou2_dy_c, rhou2_dy_r, rhou2_dy_rr;
		rhou2_dy_ll = irhou2_c[dy_ll];
		rhou2_dy_l = irhou2_c[dy_l];
		rhou2_dy_c = irhou2_c[gidx];
		rhou2_dy_r = irhou2_c[dy_r];
		rhou2_dy_rr = irhou2_c[dy_rr];

		double rho_dy_ll, rho_dy_l, rho_dy_c, rho_dy_r, rho_dy_rr;
		rho_dy_ll = irho_c[dy_ll];
		rho_dy_l = irho_c[dy_l];
		rho_dy_c = irho_c[gidx];
		rho_dy_r = irho_c[dy_r];
		rho_dy_rr = irho_c[dy_rr];

		double u2_dy_ll, u2_dy_l, u2_dy_c, u2_dy_r, u2_dy_rr;
		u2_dy_ll = rhou2_dy_ll / rho_dy_ll;
		u2_dy_l = rhou2_dy_l / rho_dy_l;
		u2_dy_c = rhou2_dy_c / rho_dy_c;
		u2_dy_r = rhou2_dy_r / rho_dy_r;
		u2_dy_rr = rhou2_dy_rr / rho_dy_rr;

		tmp0 += dns_pDer2(u2_dy_ll, u2_dy_l, u2_dy_c, u2_dy_r, u2_dy_rr, DY);

		double rhou2_dz_ll, rhou2_dz_l, rhou2_dz_c, rhou2_dz_r, rhou2_dz_rr;
		rhou2_dz_ll = irhou2_c[dz_ll];
		rhou2_dz_l = irhou2_c[dz_l];
		rhou2_dz_c = irhou2_c[gidx];
		rhou2_dz_r = irhou2_c[dz_r];
		rhou2_dz_rr = irhou2_c[dz_rr];

		double rho_dz_ll, rho_dz_l, rho_dz_c, rho_dz_r, rho_dz_rr;
		rho_dz_ll = irho_c[dz_ll];
		rho_dz_l = irho_c[dz_l];
		rho_dz_c = irho_c[gidx];
		rho_dz_r = irho_c[dz_r];
		rho_dz_rr = irho_c[dz_rr];

		double u2_dz_ll, u2_dz_l, u2_dz_c, u2_dz_r, u2_dz_rr;
		u2_dz_ll = rhou2_dz_ll / rho_dz_ll;
		u2_dz_l = rhou2_dz_l / rho_dz_l;
		u2_dz_c = rhou2_dz_c / rho_dz_c;
		u2_dz_r = rhou2_dz_r / rho_dz_r;
		u2_dz_rr = rhou2_dz_rr / rho_dz_rr;

		tmp0 += 4./3. * dns_pDer2(u2_dz_ll, u2_dz_l, u2_dz_c, u2_dz_r, u2_dz_rr, DZ);

		double rhou2_dx_ll, rhou2_dx_l, rhou2_dx_c, rhou2_dx_r, rhou2_dx_rr;
		rhou2_dx_ll = irhou2_ll[gidx];
		rhou2_dx_l = irhou2_l[gidx];
		rhou2_dx_c = irhou2_c[gidx];
		rhou2_dx_r = irhou2_r[gidx];
		rhou2_dx_rr = irhou2_rr[gidx];

		double rho_dx_ll, rho_dx_l, rho_dx_c, rho_dx_r, rho_dx_rr;
		rho_dx_ll = irho_ll[gidx];
		rho_dx_l = irho_l[gidx];
		rho_dx_c = irho_c[gidx];
		rho_dx_r = irho_r[gidx];
		rho_dx_rr = irho_rr[gidx];

		double u2_dx_ll, u2_dx_l, u2_dx_c, u2_dx_r, u2_dx_rr;
		u2_dx_ll = rhou2_dx_ll / rho_dx_ll;
		u2_dx_l = rhou2_dx_l / rho_dx_l;
		u2_dx_c = rhou2_dx_c / rho_dx_c;
		u2_dx_r = rhou2_dx_r / rho_dx_r;
		u2_dx_rr = rhou2_dx_rr / rho_dx_rr;

		//odrhou2dx[gidx] = dns_pDer1(rhou2_dx_ll, rhou2_dx_l, rhou2_dx_r, rhou2_dx_rr, DX);
		odu2dx[gidx]= dns_pDer1(u2_dx_ll, u2_dx_l, u2_dx_r, u2_dx_rr, DX);
		tmp0 += dns_pDer2(u2_dx_ll, u2_dx_l, u2_dx_c, u2_dx_r, u2_dx_rr, DX);

		tmp_du2d2xi[gidx] += tmp0;


		double u2_dx_ll_dz_ll = irhou2_ll[dz_ll] / irho_ll[dz_ll];
		double u2_dx_l_dz_ll = irhou2_l[dz_ll] / irho_l[dz_ll];
		double u2_dx_r_dz_ll = irhou2_r[dz_ll] / irho_r[dz_ll];
		double u2_dx_rr_dz_ll = irhou2_rr[dz_ll] / irho_rr[dz_ll];

		double du2dx_dz_ll = dns_pDer1(u2_dx_ll_dz_ll, u2_dx_l_dz_ll, u2_dx_r_dz_ll, u2_dx_rr_dz_ll, DX);

		double u2_dx_ll_dz_l = irhou2_ll[dz_l] / irho_ll[dz_l];
		double u2_dx_l_dz_l = irhou2_l[dz_l] / irho_l[dz_l];
		double u2_dx_r_dz_l = irhou2_r[dz_l] / irho_r[dz_l];
		double u2_dx_rr_dz_l = irhou2_rr[dz_l] / irho_rr[dz_l];

		double du2dx_dz_l = dns_pDer1(u2_dx_ll_dz_l, u2_dx_l_dz_l, u2_dx_r_dz_l, u2_dx_rr_dz_l, DX);

		double u2_dx_ll_dz_r = irhou2_ll[dz_r] / irho_ll[dz_r];
		double u2_dx_l_dz_r = irhou2_l[dz_r] / irho_l[dz_r];
		double u2_dx_r_dz_r = irhou2_r[dz_r] / irho_r[dz_r];
		double u2_dx_rr_dz_r = irhou2_rr[dz_r] / irho_rr[dz_r];

		double du2dx_dz_r = dns_pDer1(u2_dx_ll_dz_r, u2_dx_l_dz_r, u2_dx_r_dz_r, u2_dx_rr_dz_r, DX);

		double u2_dx_ll_dz_rr = irhou2_ll[dz_rr] / irho_ll[dz_rr];
		double u2_dx_l_dz_rr = irhou2_l[dz_rr] / irho_l[dz_rr];
		double u2_dx_r_dz_rr = irhou2_r[dz_rr] / irho_r[dz_rr];
		double u2_dx_rr_dz_rr = irhou2_rr[dz_rr] / irho_rr[dz_rr];

		double du2dx_dz_rr = dns_pDer1(u2_dx_ll_dz_rr, u2_dx_l_dz_rr, u2_dx_r_dz_rr, u2_dx_rr_dz_rr, DX);

		tmp_du0d2xi[gidx] += 1./3. * dns_pDer1(du2dx_dz_ll, du2dx_dz_l, du2dx_dz_r, du2dx_dz_rr, DZ);


		// Calculate idx with periodic boundary condition
		int32_t dy_ll_dz_ll, dy_ll_dz_l, dy_ll_dz_r, dy_ll_dz_rr;
		IDX((NY+Y-2)%NY, (NZ+Z-2)%NZ, dy_ll_dz_ll, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z-1)%NZ, dy_ll_dz_l, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z+1)%NZ, dy_ll_dz_r, NZ);
		IDX((NY+Y-2)%NY, (NZ+Z+2)%NZ, dy_ll_dz_rr, NZ);

		double u2_dz_ll_dy_ll = irhou2_c[dy_ll_dz_ll] / irho_c[dy_ll_dz_ll];
		double u2_dz_l_dy_ll = irhou2_c[dy_ll_dz_l] / irho_c[dy_ll_dz_l];
		double u2_dz_r_dy_ll = irhou2_c[dy_ll_dz_r] / irho_c[dy_ll_dz_r];
		double u2_dz_rr_dy_ll = irhou2_c[dy_ll_dz_rr] / irho_c[dy_ll_dz_rr];

		double du2dz_dy_ll = dns_pDer1(u2_dz_ll_dy_ll, u2_dz_l_dy_ll, u2_dz_r_dy_ll, u2_dz_rr_dy_ll, DZ);


		int32_t dy_l_dz_ll, dy_l_dz_l, dy_l_dz_r, dy_l_dz_rr;
		IDX((NY+Y-1)%NY, (NZ+Z-2)%NZ, dy_l_dz_ll, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z-1)%NZ, dy_l_dz_l, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z+1)%NZ, dy_l_dz_r, NZ);
		IDX((NY+Y-1)%NY, (NZ+Z+2)%NZ, dy_l_dz_rr, NZ);

		double u2_dz_ll_dy_l = irhou2_c[dy_l_dz_ll] / irho_c[dy_l_dz_ll];
		double u2_dz_l_dy_l = irhou2_c[dy_l_dz_l] / irho_c[dy_l_dz_l];
		double u2_dz_r_dy_l = irhou2_c[dy_l_dz_r] / irho_c[dy_l_dz_r];
		double u2_dz_rr_dy_l = irhou2_c[dy_l_dz_rr] / irho_c[dy_l_dz_rr];

		double du2dz_dy_l = dns_pDer1(u2_dz_ll_dy_l, u2_dz_l_dy_l, u2_dz_r_dy_l, u2_dz_rr_dy_l, DZ);


		int32_t dy_r_dz_ll, dy_r_dz_l, dy_r_dz_r, dy_r_dz_rr;
		IDX((NY+Y+1)%NY, (NZ+Z-2)%NZ, dy_r_dz_ll, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z-1)%NZ, dy_r_dz_l, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z+1)%NZ, dy_r_dz_r, NZ);
		IDX((NY+Y+1)%NY, (NZ+Z+2)%NZ, dy_r_dz_rr, NZ);

		double u2_dz_ll_dy_r = irhou2_c[dy_r_dz_ll] / irho_c[dy_r_dz_ll];
		double u2_dz_l_dy_r = irhou2_c[dy_r_dz_l] / irho_c[dy_r_dz_l];
		double u2_dz_r_dy_r = irhou2_c[dy_r_dz_r] / irho_c[dy_r_dz_r];
		double u2_dz_rr_dy_r = irhou2_c[dy_r_dz_rr] / irho_c[dy_r_dz_rr];

		double du2dz_dy_r = dns_pDer1(u2_dz_ll_dy_r, u2_dz_l_dy_r, u2_dz_r_dy_r, u2_dz_rr_dy_r, DZ);


		int32_t dy_rr_dz_ll, dy_rr_dz_l, dy_rr_dz_r, dy_rr_dz_rr;
		IDX((NY+Y+2)%NY, (NZ+Z-2)%NZ, dy_rr_dz_ll, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z-1)%NZ, dy_rr_dz_l, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z+1)%NZ, dy_rr_dz_r, NZ);
		IDX((NY+Y+2)%NY, (NZ+Z+2)%NZ, dy_rr_dz_rr, NZ);

		double u2_dz_ll_dy_rr = irhou2_c[dy_rr_dz_ll] / irho_c[dy_rr_dz_ll];
		double u2_dz_l_dy_rr = irhou2_c[dy_rr_dz_l] / irho_c[dy_rr_dz_l];
		double u2_dz_r_dy_rr = irhou2_c[dy_rr_dz_r] / irho_c[dy_rr_dz_r];
		double u2_dz_rr_dy_rr = irhou2_c[dy_rr_dz_rr] / irho_c[dy_rr_dz_rr];

		double du2dz_dy_rr = dns_pDer1(u2_dz_ll_dy_rr, u2_dz_l_dy_rr, u2_dz_r_dy_rr, u2_dz_rr_dy_rr, DZ);

		tmp_du1d2xi[gidx] += 1./3. * dns_pDer1(du2dz_dy_ll, du2dz_dy_l, du2dz_dy_r, du2dz_dy_rr, DY);
	}
}


__global__ void dns_drhoETpdyz(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ irho_c,
	/*order 0*/ double * __restrict__ irhou0_c,
	/*order 0*/ double * __restrict__ irhou1_c,
	/*order 0*/ double * __restrict__ irhou2_c,
	/*order 0*/ double * __restrict__ irhoE_c,
	/*order 0*/ double * __restrict__ idu0dx,
	/*order 0*/ double * __restrict__ idu1dx,
	/*order 0*/ double * __restrict__ idu2dx,
	int sy_bc_ll, int sy_bc_l, int sy_bc_r, int sy_bc_rr,
	/*order 0*/ double * __restrict__ Res_rho,
	/*order 0*/ double * __restrict__ Res_rhou0,
	/*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ Res_rhou2,
	/*order 0*/ double * __restrict__ Res_rhoE) {

	__shared__ double s_rho_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo
	__shared__ double s_invrho_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo
	__shared__ double s_rhoE_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo

	__shared__ double s_u0_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo
	__shared__ double s_u1_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo
	__shared__ double s_u2_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo

	__shared__ double s_p_c[BLOCKSIZE_Y+4][BLOCKSIZE_Z+4]; // 4-wide halo


	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t gidx;
	// Idx in Block
	int32_t cb, rb;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &cb, &rb);
	gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &cb, &rb);

	double tmp0 = 1. / (MINF * MINF * PR * RE * (GAMA - 1));


	if (gidx<block_ncc) {
		rb+=2;
		cb+=2;

		s_rho_c[rb][cb] = irho_c[gidx];
		s_invrho_c[rb][cb] = 1./s_rho_c[rb][cb];
		s_rhoE_c[rb][cb] = irhoE_c[gidx];

		s_u0_c[rb][cb] = irhou0_c[gidx] / s_rho_c[rb][cb];
		s_u1_c[rb][cb] = irhou1_c[gidx] / s_rho_c[rb][cb];
		s_u2_c[rb][cb] = irhou2_c[gidx] / s_rho_c[rb][cb];

		s_p_c[rb][cb] = calp_ui(s_rhoE_c[rb][cb], s_rho_c[rb][cb], s_u0_c[rb][cb], s_u1_c[rb][cb], s_u2_c[rb][cb]);


		int32_t Y, Z;
		int32_t dy_ll, dy_l, dy_r, dy_rr;
		int32_t dz_ll, dz_l, dz_r, dz_rr;
		COORDS(gidx, Y, Z, NZ);

		// get halos
		if (rb < 4) {
			IDX((NY+Y-2)%NY, Z, dy_ll, NZ);
			s_rho_c[rb-2][cb] = irho_c[dy_ll];
			s_invrho_c[rb-2][cb] = 1./s_rho_c[rb-2][cb];
			s_rhoE_c[rb-2][cb] = irhoE_c[dy_ll];

			s_u0_c[rb-2][cb] = irhou0_c[dy_ll] / s_rho_c[rb-2][cb];
			s_u1_c[rb-2][cb] = irhou1_c[dy_ll] / s_rho_c[rb-2][cb];
			s_u2_c[rb-2][cb] = irhou2_c[dy_ll] / s_rho_c[rb-2][cb];

			s_p_c[rb-2][cb] = calp_ui(s_rhoE_c[rb-2][cb], s_rho_c[rb-2][cb], s_u0_c[rb-2][cb], s_u1_c[rb-2][cb], s_u2_c[rb-2][cb]);

		}
		if (rb >= BLOCKSIZE_Y) {
			IDX((NY+Y+2)%NY, Z, dy_rr, NZ);
			s_rho_c[rb+2][cb] = irho_c[dy_rr];
			s_invrho_c[rb+2][cb] = 1./s_rho_c[rb+2][cb];
			s_rhoE_c[rb+2][cb] = irhoE_c[dy_rr];

			s_u0_c[rb+2][cb] = irhou0_c[dy_rr] / s_rho_c[rb+2][cb];
			s_u1_c[rb+2][cb] = irhou1_c[dy_rr] / s_rho_c[rb+2][cb];
			s_u2_c[rb+2][cb] = irhou2_c[dy_rr] / s_rho_c[rb+2][cb];

			s_p_c[rb+2][cb] = calp_ui(s_rhoE_c[rb+2][cb], s_rho_c[rb+2][cb], s_u0_c[rb+2][cb], s_u1_c[rb+2][cb], s_u2_c[rb+2][cb]);

		}

		// get halos
		if (cb < 4) {
			IDX(Y, (NZ+Z-2)%NZ, dz_ll, NZ);
			s_rho_c[rb][cb-2] = irho_c[dz_ll];
			s_invrho_c[rb][cb-2] = 1./s_rho_c[rb][cb-2];
			s_rhoE_c[rb][cb-2] = irhoE_c[dz_ll];

			s_u0_c[rb][cb-2] = irhou0_c[dz_ll] / s_rho_c[rb][cb-2];
			s_u1_c[rb][cb-2] = irhou1_c[dz_ll] / s_rho_c[rb][cb-2];
			s_u2_c[rb][cb-2] = irhou2_c[dz_ll] / s_rho_c[rb][cb-2];

			s_p_c[rb][cb-2] = calp_ui(s_rhoE_c[rb][cb-2], s_rho_c[rb][cb-2], s_u0_c[rb][cb-2], s_u1_c[rb][cb-2], s_u2_c[rb][cb-2]);

		}
		if (cb >= BLOCKSIZE_Z) {
			IDX(Y, (NZ+Z+2)%NZ, dz_rr, NZ);
			s_rho_c[rb][cb+2] = irho_c[dz_rr];
			s_invrho_c[rb][cb+2] = 1./s_rho_c[rb][cb+2];
			s_rhoE_c[rb][cb+2] = irhoE_c[dz_rr];

			s_u0_c[rb][cb+2] = irhou0_c[dz_rr] / s_rho_c[rb][cb+2];
			s_u1_c[rb][cb+2] = irhou1_c[dz_rr] / s_rho_c[rb][cb+2];
			s_u2_c[rb][cb+2] = irhou2_c[dz_rr] / s_rho_c[rb][cb+2];

			s_p_c[rb][cb+2] = calp_ui(s_rhoE_c[rb][cb+2], s_rho_c[rb][cb+2], s_u0_c[rb][cb+2], s_u1_c[rb][cb+2], s_u2_c[rb][cb+2]);


		}

		__syncthreads();

		double tmp_Res_rho = 0;
		double tmp_Res_rhou0 = 0;
		double tmp_Res_rhou1 = 0;
		double tmp_Res_rhou2 = 0;
		double tmp_Res_rhoE = 0;
		
		double tmp_dTd2xi = 0;
		double tmp1;
		double tmp2;
		double tmp3;
		double tmp4;

		tmp1 = idu0dx[gidx];
		tmp2 = dns_pDer1(s_u1_c[rb-2][cb], s_u1_c[rb-1][cb], s_u1_c[rb+1][cb], s_u1_c[rb+2][cb], DY);
		tmp3 = dns_pDer1(s_u2_c[rb][cb-2], s_u2_c[rb][cb-1], s_u2_c[rb][cb+1], s_u2_c[rb][cb+2], DZ);

		tmp4 = -0.5 * (tmp1 + tmp2 + tmp3);
		tmp_Res_rho += tmp4 * s_rho_c[rb][cb];
		tmp_Res_rhou0 += tmp4 * s_rho_c[rb][cb] * s_u0_c[rb][cb];
		tmp_Res_rhou1 += tmp4 * s_rho_c[rb][cb] * s_u1_c[rb][cb];
		tmp_Res_rhou2 += tmp4 * s_rho_c[rb][cb] * s_u2_c[rb][cb];
		tmp_Res_rhoE += tmp4 * s_rhoE_c[rb][cb];

		tmp_Res_rhoE += 1./RE * (-2./3. * tmp1 - 2./3. * tmp2 + 4./3. * tmp3) * tmp3;
		tmp_Res_rhoE += 1./RE * (-2./3. * tmp1 + 4./3. * tmp2 - 2./3. * tmp3) * tmp2;
		tmp_Res_rhoE += 1./RE * ( 4./3. * tmp1 - 2./3. * tmp2 - 2./3. * tmp3) * tmp1;

		tmp1 = idu1dx[gidx];
		tmp2 = dns_pDer1(s_u0_c[rb-2][cb], s_u0_c[rb-1][cb], s_u0_c[rb+1][cb], s_u0_c[rb+2][cb], DY);
		tmp_Res_rhoE += 1./RE * (tmp2 + tmp1) * tmp2 + 1./RE * (tmp2 + tmp1) * tmp1;

		tmp3 = idu2dx[gidx];
		tmp4 = dns_pDer1(s_u0_c[rb][cb-2], s_u0_c[rb][cb-1], s_u0_c[rb][cb+1], s_u0_c[rb][cb+2], DZ);
		tmp_Res_rhoE += 1./RE * (tmp4 + tmp3) * tmp4 + 1./RE * (tmp4 + tmp3) * tmp3;

		
		tmp_Res_rho += -0.5 * dns_pDer1(s_rho_c[rb-2][cb], s_rho_c[rb-1][cb], s_rho_c[rb+1][cb], s_rho_c[rb+2][cb], DY) * s_u1_c[rb][cb];
		tmp_Res_rhou0 += -0.5 * dns_pDer1(s_rho_c[rb-2][cb] * s_u0_c[rb-2][cb] * s_u1_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u0_c[rb-1][cb] * s_u1_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u0_c[rb+1][cb] * s_u1_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u0_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp_Res_rhou2 += -0.5 * dns_pDer1(s_rho_c[rb-2][cb] * s_u2_c[rb-2][cb] * s_u1_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u2_c[rb-1][cb] * s_u1_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u2_c[rb+1][cb] * s_u1_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u2_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp_Res_rhou0 += -0.5 * dns_pDer1(s_rho_c[rb-2][cb] * s_u0_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u0_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u0_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u0_c[rb+2][cb], DY) * s_u1_c[rb][cb];
		tmp_Res_rhou2 += -0.5 * dns_pDer1(s_rho_c[rb-2][cb] * s_u2_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u2_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u2_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u2_c[rb+2][cb], DY) * s_u1_c[rb][cb];
		tmp_Res_rhou1 += -0.5 * dns_pDer1(s_rho_c[rb-2][cb] * s_u1_c[rb-2][cb] * s_u1_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u1_c[rb-1][cb] * s_u1_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u1_c[rb+1][cb] * s_u1_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u1_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp1 = dns_pDer1(s_rho_c[rb-2][cb] * s_u1_c[rb-2][cb], s_rho_c[rb-1][cb] * s_u1_c[rb-1][cb], s_rho_c[rb+1][cb] * s_u1_c[rb+1][cb], s_rho_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp_Res_rhou1 += -0.5 * tmp1 * s_u1_c[rb][cb];
		tmp_Res_rho += -0.5 * tmp1;

	
		tmp_Res_rho += -0.5 * dns_pDer1(s_rho_c[rb][cb-2], s_rho_c[rb][cb-1], s_rho_c[rb][cb+1], s_rho_c[rb][cb+2], DZ) * s_u2_c[rb][cb];
		tmp_Res_rhou0 += -0.5 * dns_pDer1(s_rho_c[rb][cb-2] * s_u0_c[rb][cb-2] * s_u2_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u0_c[rb][cb-1] * s_u2_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u0_c[rb][cb+1] * s_u2_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u0_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		tmp_Res_rhou1 += -0.5 * dns_pDer1(s_rho_c[rb][cb-2] * s_u1_c[rb][cb-2] * s_u2_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u1_c[rb][cb-1] * s_u2_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u1_c[rb][cb+1] * s_u2_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u1_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		tmp_Res_rhou0 += -0.5 * dns_pDer1(s_rho_c[rb][cb-2] * s_u0_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u0_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u0_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u0_c[rb][cb+2], DZ) * s_u2_c[rb][cb];
		tmp_Res_rhou1 += -0.5 * dns_pDer1(s_rho_c[rb][cb-2] * s_u1_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u1_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u1_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u1_c[rb][cb+2], DZ) * s_u2_c[rb][cb];
		tmp_Res_rhou2 += -0.5 * dns_pDer1(s_rho_c[rb][cb-2] * s_u2_c[rb][cb-2] * s_u2_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u2_c[rb][cb-1] * s_u2_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u2_c[rb][cb+1] * s_u2_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u2_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		tmp1 = dns_pDer1(s_rho_c[rb][cb-2] * s_u2_c[rb][cb-2], s_rho_c[rb][cb-1] * s_u2_c[rb][cb-1], s_rho_c[rb][cb+1] * s_u2_c[rb][cb+1], s_rho_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		tmp_Res_rhou2 += -0.5 * tmp1 * s_u2_c[rb][cb];
		tmp_Res_rho += -0.5 * tmp1;
		

		tmp1 = dns_pDer1(s_u1_c[rb][cb-2], s_u1_c[rb][cb-1], s_u1_c[rb][cb+1], s_u1_c[rb][cb+2], DZ);
		tmp2 = dns_pDer1(s_u2_c[rb-2][cb], s_u2_c[rb-1][cb], s_u2_c[rb+1][cb], s_u2_c[rb+2][cb], DY);
		tmp_Res_rhoE += 1./RE * (tmp1 + tmp2) * tmp1 + 1./RE * (tmp1 + tmp2) * tmp2;

		tmp_Res_rhou1 += -dns_pDer1(s_p_c[rb-2][cb], s_p_c[rb-1][cb], s_p_c[rb+1][cb], s_p_c[rb+2][cb], DY);
		tmp_Res_rhoE -= dns_pDer1(s_p_c[rb-2][cb] * s_u1_c[rb-2][cb], s_p_c[rb-1][cb] * s_u1_c[rb-1][cb], s_p_c[rb+1][cb] * s_u1_c[rb+1][cb], s_p_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp_dTd2xi  += dns_pDer2(calT_ui(s_p_c[rb-2][cb], s_invrho_c[rb-2][cb]), calT_ui(s_p_c[rb-1][cb], s_invrho_c[rb-1][cb]), calT_ui(s_p_c[rb][cb], s_invrho_c[rb][cb]), calT_ui(s_p_c[rb+1][cb], s_invrho_c[rb+1][cb]), calT_ui(s_p_c[rb+2][cb], s_invrho_c[rb+2][cb]), DY);
		
		tmp_Res_rhou2 += -dns_pDer1(s_p_c[rb][cb-2], s_p_c[rb][cb-1], s_p_c[rb][cb+1], s_p_c[rb][cb+2], DZ);
		tmp_Res_rhoE -= dns_pDer1(s_p_c[rb][cb-2]* s_u2_c[rb][cb-2], s_p_c[rb][cb-1] * s_u2_c[rb][cb-1], s_p_c[rb][cb+1] * s_u2_c[rb][cb+1], s_p_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		tmp_dTd2xi  +=  dns_pDer2(calT_ui(s_p_c[rb][cb-2], s_invrho_c[rb][cb-2]), calT_ui(s_p_c[rb][cb-1], s_invrho_c[rb][cb-1]), calT_ui(s_p_c[rb][cb], s_invrho_c[rb][cb]), calT_ui(s_p_c[rb][cb+1], s_invrho_c[rb][cb+1]), calT_ui(s_p_c[rb][cb+2], s_invrho_c[rb][cb+2]), DZ);	

		tmp_Res_rhoE += -0.5 * dns_pDer1(s_rhoE_c[rb-2][cb], s_rhoE_c[rb-1][cb], s_rhoE_c[rb+1][cb], s_rhoE_c[rb+2][cb], DY) * s_u1_c[rb][cb];
		tmp_Res_rhoE += -0.5 * dns_pDer1(s_rhoE_c[rb-2][cb] * s_u1_c[rb-2][cb], s_rhoE_c[rb-1][cb] * s_u1_c[rb-1][cb], s_rhoE_c[rb+1][cb] * s_u1_c[rb+1][cb], s_rhoE_c[rb+2][cb] * s_u1_c[rb+2][cb], DY);
		tmp_Res_rhoE += -0.5 * dns_pDer1(s_rhoE_c[rb][cb-2], s_rhoE_c[rb][cb-1], s_rhoE_c[rb][cb+1], s_rhoE_c[rb][cb+2], DZ) * s_u2_c[rb][cb];
		tmp_Res_rhoE += -0.5 * dns_pDer1(s_rhoE_c[rb][cb-2] * s_u2_c[rb][cb-2], s_rhoE_c[rb][cb-1] * s_u2_c[rb][cb-1], s_rhoE_c[rb][cb+1] * s_u2_c[rb][cb+1], s_rhoE_c[rb][cb+2] * s_u2_c[rb][cb+2], DZ);
		
		tmp_Res_rhoE += tmp_dTd2xi * tmp0;

		Res_rho[gidx] = tmp_Res_rho;
		Res_rhou0[gidx] = tmp_Res_rhou0;
		Res_rhou1[gidx] = tmp_Res_rhou1;
		Res_rhou2[gidx] = tmp_Res_rhou2;
		Res_rhoE[gidx] = tmp_Res_rhoE;
			
	}

}

__global__ void dns_drhoETpdx(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ irho_c,
	/*order 1*/ double * __restrict__ irho_l, double * __restrict__ irho_r,
	/*order 2*/ double * __restrict__ irho_ll, double * __restrict__ irho_rr,
	/*order 0*/ double * __restrict__ irhou0_c,
	/*order 1*/ double * __restrict__ irhou0_l, double * __restrict__ irhou0_r,
	/*order 2*/ double * __restrict__ irhou0_ll, double * __restrict__ irhou0_rr,
	/*order 0*/ double * __restrict__ irhou1_c,
	/*order 1*/ double * __restrict__ irhou1_l, double * __restrict__ irhou1_r,
	/*order 2*/ double * __restrict__ irhou1_ll, double * __restrict__ irhou1_rr,
	/*order 0*/ double * __restrict__ irhou2_c,
	/*order 1*/ double * __restrict__ irhou2_l, double * __restrict__ irhou2_r,
	/*order 2*/ double * __restrict__ irhou2_ll, double * __restrict__ irhou2_rr,
	/*order 0*/ double * __restrict__ irhoE_c,
	/*order 1*/ double * __restrict__ irhoE_l, double * __restrict__ irhoE_r,
	/*order 2*/ double * __restrict__ irhoE_ll, double * __restrict__ irhoE_rr,
	int sy_bc_ll, int sy_bc_l, int sy_bc_r, int sy_bc_rr,
	/*order 0*/ double * __restrict__ Res_rho,
	/*order 0*/ double * __restrict__ Res_rhou0,
	/*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ Res_rhou2,
	/*order 0*/ double * __restrict__ Res_rhoE) {

	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t gidx;
	// Idx in Block
	int32_t cb, rb;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &cb, &rb);
	gidx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &cb, &rb);

	double tmp0 = 1. / (MINF * MINF * PR * RE * (GAMA - 1));

	double fac_ll = 1./12.;
	double fac_l = -2./3.;
	double fac_r = 2./3.;
	double fac_rr = -1./12.;

	double fac_dk = 1./DX;

	if (gidx<block_ncc) {

		
		double rho, invrho, rhou0, u0, rhou1, rhou2, rhoE, p;
		double drhodx;
		double dpdx, dpu0dx;
		double drhoEdx, drhoEu0dx;
		double drhou0u0dx, drhou1u0dx, drhou2u0dx;
		double dTd2x;
		double drhou0dx, drhou1dx, drhou2dx;

		// dx_ll
		rho = irho_ll[gidx];
		invrho = 1./rho;
		rhou0 = irhou0_ll[gidx];
		u0 = rhou0 * invrho;
		rhou1 = irhou1_ll[gidx];
		rhou2 = irhou2_ll[gidx];
		rhoE = irhoE_ll[gidx];
		p = calp_ui(rhoE, rho, u0 , rhou1 * invrho, rhou2 * invrho);

		drhodx = fac_ll * rho;
		dpdx = fac_ll * p;
		dpu0dx = fac_ll * p * sy_bc_ll * u0;
		drhoEdx = fac_ll * rhoE;
		drhoEu0dx = fac_ll * rhoE * sy_bc_ll * u0;
		drhou1u0dx = fac_ll * rhou1 * sy_bc_ll * u0;
		drhou2u0dx = fac_ll * rhou2 * sy_bc_ll * u0;
		drhou0u0dx = fac_ll * sy_bc_ll * rhou0 * sy_bc_ll * u0;

		drhou0dx = fac_ll * sy_bc_ll * rhou0;
		drhou1dx = fac_ll * rhou1;
		drhou2dx = fac_ll * rhou2;

		dTd2x = -fac_ll * calT_ui(p, invrho);

		// dx_l
		rho = irho_l[gidx];
		invrho = 1./rho;
		rhou0 = irhou0_l[gidx];
		u0 = rhou0 * invrho;
		rhou1 = irhou1_l[gidx];
		rhou2 = irhou2_l[gidx];
		rhoE = irhoE_l[gidx];
		p = calp_ui(rhoE, rho, u0 , rhou1 * invrho, rhou2 * invrho);

		drhodx += fac_l * rho;
		dpdx += fac_l * p;
		dpu0dx += fac_l* p * sy_bc_l * u0;
		drhoEdx += fac_l * rhoE;
		drhoEu0dx += fac_l * rhoE * sy_bc_l * u0;
		drhou1u0dx += fac_l * rhou1 * sy_bc_l * u0;
		drhou2u0dx += fac_l * rhou2 * sy_bc_l * u0;
		drhou0u0dx += fac_l * sy_bc_l * rhou0 * sy_bc_l * u0;

		drhou0dx += fac_l * sy_bc_l * rhou0;
		drhou1dx += fac_l * rhou1;
		drhou2dx += fac_l * rhou2;

		dTd2x += -2. * fac_l * calT_ui(p, invrho);

		// dx_r
		rho = irho_r[gidx];
		invrho = 1./rho;
		rhou0 = irhou0_r[gidx];
		u0 = rhou0 * invrho;
		rhou1 = irhou1_r[gidx];
		rhou2 = irhou2_r[gidx];
		rhoE = irhoE_r[gidx];
		p = calp_ui(rhoE, rho, u0 , rhou1 * invrho, rhou2 * invrho);

		drhodx += fac_r * rho;
		dpdx += fac_r * p;
		dpu0dx += fac_r* p * sy_bc_r * u0;
		drhoEdx += fac_r * rhoE;
		drhoEu0dx += fac_r * rhoE * sy_bc_r * u0;
		drhou1u0dx += fac_r * rhou1 * sy_bc_r * u0;
		drhou2u0dx += fac_r * rhou2 * sy_bc_r * u0;
		drhou0u0dx += fac_r * sy_bc_r * rhou0 * sy_bc_r * u0;

		drhou0dx += fac_r * sy_bc_r * rhou0;
		drhou1dx += fac_r * rhou1;
		drhou2dx += fac_r * rhou2;

		dTd2x += 2. * fac_r * calT_ui(p, invrho);

		// dx_rr
		rho = irho_rr[gidx];
		invrho = 1./rho;
		rhou0 = irhou0_rr[gidx];
		u0 = rhou0 * invrho;
		rhou1 = irhou1_rr[gidx];
		rhou2 = irhou2_rr[gidx];
		rhoE = irhoE_rr[gidx];
		p = calp_ui(rhoE, rho, u0 , rhou1 * invrho, rhou2 * invrho);

		drhodx += fac_rr * rho;
		dpdx += fac_rr * p;
		dpu0dx += fac_rr* p * sy_bc_rr * u0;
		drhoEdx += fac_rr * rhoE;
		drhoEu0dx += fac_rr * rhoE * sy_bc_rr * u0;
		drhou1u0dx += fac_rr * rhou1 * sy_bc_rr * u0;
		drhou2u0dx += fac_rr * rhou2 * sy_bc_rr * u0;
		drhou0u0dx += fac_rr * sy_bc_rr * rhou0 * sy_bc_rr * u0;

		drhou0dx += fac_rr * sy_bc_rr * rhou0;
		drhou1dx += fac_rr * rhou1;
		drhou2dx += fac_rr * rhou2;

		dTd2x += fac_rr * calT_ui(p, invrho);

		// dx_c
		rho = irho_c[gidx];
		invrho = 1./rho;
		rhou0 = irhou0_c[gidx];
		u0 = rhou0 * invrho;
		rhou1 = irhou1_c[gidx];
		rhou2 = irhou2_c[gidx];
		rhoE = irhoE_c[gidx];
		p = calp_ui(rhoE, rho, u0 , rhou1 * invrho, rhou2 * invrho);

		dTd2x += -5./2. * calT_ui(p, invrho);

		double tmp_Res_rho = 0;
		tmp_Res_rho += -0.5 * fac_dk * drhodx * u0;
		tmp_Res_rho += -0.5 * fac_dk * drhou0dx;

		double tmp_Res_rhou0 = 0;
		tmp_Res_rhou0 += - 0.5 * fac_dk * drhou0u0dx;
		tmp_Res_rhou0 += - fac_dk * dpdx;
		tmp_Res_rhou0 += -0.5 * fac_dk * drhou0dx * u0;

		double tmp_Res_rhou1 = 0;
		tmp_Res_rhou1 += -0.5 * fac_dk * drhou1u0dx;
		tmp_Res_rhou1 += -0.5 * fac_dk * drhou1dx * u0;

		double tmp_Res_rhou2 = 0;
		tmp_Res_rhou2 += -0.5 * fac_dk * drhou2u0dx;
		tmp_Res_rhou2 += -0.5 * fac_dk * drhou2dx * u0;

		double tmp_Res_rhoE = 0;
		tmp_Res_rhoE -= fac_dk * dpu0dx;
		tmp_Res_rhoE += -0.5 * fac_dk * drhoEdx * u0;
		tmp_Res_rhoE += -0.5 * fac_dk * drhoEu0dx;
		tmp_Res_rhoE += fac_dk * fac_dk * dTd2x * tmp0 ;

		Res_rho[gidx] += tmp_Res_rho;
		Res_rhou0[gidx] += tmp_Res_rhou0;
		Res_rhou1[gidx] += tmp_Res_rhou1;
		Res_rhou2[gidx] += tmp_Res_rhou2;
		Res_rhoE[gidx] += tmp_Res_rhoE;
	}

}

__global__ void dns_Res_StageAdvance(int32_t i_worker, int32_t order_in, int32_t order_out,
	/*order 0*/ double * __restrict__ rho,
	/*order 0*/ double * __restrict__ rhou0,
	/*order 0*/ double * __restrict__ rhou1,
	/*order 0*/ double * __restrict__ rhou2,
	/*order 0*/ double * __restrict__ rhoE,
	
	/*order 0*/ double * __restrict__ irho_old,
	/*order 0*/ double * __restrict__ irhou0_old,
	/*order 0*/ double * __restrict__ irhou1_old,
	/*order 0*/ double * __restrict__ irhou2_old,
	/*order 0*/ double * __restrict__ irhoE_old,
	
	/*order 0*/ double * __restrict__ orho,
	/*order 0*/ double * __restrict__ orhou0,
	/*order 0*/ double * __restrict__ orhou1,
	/*order 0*/ double * __restrict__ orhou2,
	/*order 0*/ double * __restrict__ orhoE,

	/*order 0*/ double * __restrict__ orho_old,
	/*order 0*/ double * __restrict__ orhou0_old,
	/*order 0*/ double * __restrict__ orhou1_old,
	/*order 0*/ double * __restrict__ orhou2_old,
	/*order 0*/ double * __restrict__ orhoE_old,
	
	double rknew, double rkold,
	/*order 0*/ double * __restrict__ Res_rho,
	/*order 0*/ double * __restrict__ Res_rhou0,
	/*order 0*/ double * __restrict__ Res_rhou1,
	/*order 0*/ double * __restrict__ Res_rhou2,
	/*order 0*/ double * __restrict__ Res_rhoE,
	/*order 0*/ double * __restrict__ tmp_du0d2xi,
	/*order 0*/ double * __restrict__ tmp_du1d2xi,
	/*order 0*/ double * __restrict__ tmp_du2d2xi) {


	// Thread Idx
	int32_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
	// Global Idx
	int32_t idx;
	// Idx in Block
	int32_t col_in_block, row_in_block;
	// Global Idx
	//gidx = thread_to_global_idx(problemsize, tidx, block_size_z, block_size_y, warp_size_z, warp_size_y, &col_in_block, &row_in_block);
	idx = thread_to_global_idx(my_n_part, tidx, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y, &col_in_block, &row_in_block);

	if (idx<block_ncc) {
		double lRes_rho = 0;
		double lRes_rhou0 = 0;
		double lRes_rhou1 = 0;
		double lRes_rhou2 = 0;
		double lRes_rhoE = 0;

		double tmp0;

		lRes_rho += Res_rho[idx];
		orho[idx] = DT * rknew * lRes_rho + irho_old[idx];
		orho_old[idx] = DT * rkold * lRes_rho + irho_old[idx];

		double frac0 = 1./RE;

		double lu0 = rhou0[idx] / rho[idx];
		tmp0 = frac0 * tmp_du0d2xi[idx];
		lRes_rhou0 += tmp0;
		lRes_rhoE += lu0 * tmp0;

		lRes_rhou0 += Res_rhou0[idx];
		orhou0[idx] = DT * rknew * lRes_rhou0 + irhou0_old[idx];
		orhou0_old[idx] = DT * rkold * lRes_rhou0 + irhou0_old[idx];


		double lu1 = rhou1[idx] / rho[idx];
		tmp0 = frac0 * tmp_du1d2xi[idx];
		lRes_rhou1 += tmp0;
		lRes_rhoE += lu1 * tmp0;

		lRes_rhou1 += Res_rhou1[idx];
		orhou1[idx] = DT * rknew * lRes_rhou1 + irhou1_old[idx];
		orhou1_old[idx] = DT * rkold * lRes_rhou1 + irhou1_old[idx];


		double lu2 = rhou2[idx] / rho[idx];
		tmp0 = frac0 * tmp_du2d2xi[idx];
		lRes_rhou2 += tmp0;
		lRes_rhoE += lu2 * tmp0;

		lRes_rhou2 += Res_rhou2[idx];
		orhou2[idx] = DT * rknew * lRes_rhou2 + irhou2_old[idx];
		orhou2_old[idx] = DT * rkold * lRes_rhou2 + irhou2_old[idx];

		lRes_rhoE += Res_rhoE[idx];

		orhoE[idx] = DT * rknew * lRes_rhoE + irhoE_old[idx];
		orhoE_old[idx] = DT * rkold * lRes_rhoE + irhoE_old[idx];

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



	dns_du0dxyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, my_n_part, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y,
		&p_c[offset_rho], &p_l[offset_rho], &p_r[offset_rho], &p_ll[offset_rho], &p_rr[offset_rho],
		&p_c[offset_rhou0], &p_l[offset_rhou0], &p_r[offset_rhou0], &p_ll[offset_rhou0], &p_rr[offset_rhou0],
		sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr,
		(double*) d_du0dx,
		(double*) tmp_du0d2xi, (double*) tmp_du1d2xi, (double*) tmp_du2d2xi);

	dns_du1dxyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, my_n_part, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y,
		&p_c[offset_rho], &p_l[offset_rho], &p_r[offset_rho], &p_ll[offset_rho], &p_rr[offset_rho],
		&p_c[offset_rhou1], &p_l[offset_rhou1], &p_r[offset_rhou1], &p_ll[offset_rhou1], &p_rr[offset_rhou1],
		(double*) d_du1dx,
		(double*) d_Res_rhou1,
		(double*) tmp_du0d2xi, (double*) tmp_du1d2xi, (double*) tmp_du2d2xi);

	dns_du2dxyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0, my_n_part, BLOCKSIZE_Z, BLOCKSIZE_Y, WARPSIZE_Z, WARPSIZE_Y,
		&p_c[offset_rho], &p_l[offset_rho], &p_r[offset_rho], &p_ll[offset_rho], &p_rr[offset_rho],
		&p_c[offset_rhou2], &p_l[offset_rhou2], &p_r[offset_rhou2], &p_ll[offset_rhou2], &p_rr[offset_rhou2],
		(double*) d_du2dx,
		(double*) tmp_du0d2xi, (double*) tmp_du1d2xi, (double*) tmp_du2d2xi);

	dns_drhoETpdyz <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
		&p_c[offset_rho], &p_c[offset_rhou0], &p_c[offset_rhou1], &p_c[offset_rhou2], &p_c[offset_rhoE],
		(double*) d_du0dx, (double*) d_du1dx, (double*) d_du2dx,
		sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr,
		(double*) d_Res_rho,
		(double*) d_Res_rhou0,
		(double*) d_Res_rhou1,
		(double*) d_Res_rhou2,
		(double*) d_Res_rhoE);

	dns_drhoETpdx <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
		&p_c[offset_rho], &p_l[offset_rho], &p_r[offset_rho], &p_ll[offset_rho], &p_rr[offset_rho],
		&p_c[offset_rhou0], &p_l[offset_rhou0], &p_r[offset_rhou0], &p_ll[offset_rhou0], &p_rr[offset_rhou0],
		&p_c[offset_rhou1], &p_l[offset_rhou1], &p_r[offset_rhou1], &p_ll[offset_rhou1], &p_rr[offset_rhou1],
		&p_c[offset_rhou2], &p_l[offset_rhou2], &p_r[offset_rhou2], &p_ll[offset_rhou2], &p_rr[offset_rhou2],
		&p_c[offset_rhoE], &p_l[offset_rhoE], &p_r[offset_rhoE], &p_ll[offset_rhoE], &p_rr[offset_rhoE],
		sy_bc_ll, sy_bc_l, sy_bc_r, sy_bc_rr,
		(double*) d_Res_rho,
		(double*) d_Res_rhou0,
		(double*) d_Res_rhou1,
		(double*) d_Res_rhou2,
		(double*) d_Res_rhoE);

	dns_Res_StageAdvance <<<gridSize,threads_per_block,0,*stream>>>(0, 0, 0,
			(double*) &p_c[offset_rho],
			&p_c[offset_rhou0], &p_c[offset_rhou1], &p_c[offset_rhou2], &p_c[offset_rhoE],
			&p_c[offset_rho_old],
			&p_c[offset_rhou0_old], &p_c[offset_rhou1_old], &p_c[offset_rhou2_old],
			&p_c[offset_rhoE_old],
			&p_c_out[offset_rho], 
			&p_c_out[offset_rhou0], &p_c_out[offset_rhou1], &p_c_out[offset_rhou2],
			&p_c_out[offset_rhoE],
			&p_c_out[offset_rho_old],
			&p_c_out[offset_rhou0_old], &p_c_out[offset_rhou1_old], &p_c_out[offset_rhou2_old],
			&p_c_out[offset_rhoE_old],
			rknew, rkold,
			(double*) d_Res_rho,
			(double*) d_Res_rhou0,
			(double*) d_Res_rhou1,
			(double*) d_Res_rhou2,
			(double*) d_Res_rhoE,
			(double*) tmp_du0d2xi, (double*) tmp_du1d2xi, (double*) tmp_du2d2xi);

	//// Copy Header
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
