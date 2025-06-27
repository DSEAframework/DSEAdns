#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <math.h>
#include <cmath>

// parameters for algorithm

#include "dsea_case.h"
#include "dsea_opti_case.h"

// Base Case 8 Workers:
// 6000 Timesteps for Case 64
// (6000 * 3) / 8 = 2250 Super Cycles
// 12000 Timesteps for Case 128
// (12000 * 3) / 8 = 4500 Super Cycles
// 24000 Timesteps for Case 256
// (24000 * 3) / 8 = 9000 Super Cycles


#ifndef DOUTPUT
	#define DOUTPUT -1
#endif

#define DOUTPUTSTART 0

#define block_n_fields				10

#if defined (case_64)
	#define block_ncc					(64*64)
	#define block_header_size			64
 	#define my_n_part							64UL
	#define DDT										0.003385
#elif defined (case_90)
	#define block_ncc					(90*90)
	#define block_header_size			64
 	#define my_n_part							90UL
	#define DDT										0.003385 / 1.41421356237 											
#elif defined (case_128)
	#define block_ncc					(128*128)
	#define block_header_size			64
 	#define my_n_part							128UL
	#define DDT										0.003385 / 2.
#elif defined (case_180)
	#define block_ncc					(180*180)
	#define block_header_size			64
 	#define my_n_part							180UL
	#define DDT										0.003385 / 2. / 1.41421356237 
#elif defined (case_256)
	#define block_ncc					(256*256)
	#define block_header_size			64
 	#define my_n_part							256UL
	#define DDT										0.003385 / 2. / 2.
#elif defined (case_362)
	#define block_ncc					(362*362)
	#define block_header_size			64
 	#define my_n_part							362UL
	#define DDT										0.003385 / 2. / 2. / 1.41421356237 
#elif defined (case_512)
	#define block_ncc					(512*512)
	#define block_header_size			64
 	#define my_n_part							512UL
	#define DDT										0.003385 / 2. / 2. / 2.
#elif defined (case_724)
	#define block_ncc					(724*724)
	#define block_header_size			64
 	#define my_n_part							724UL
	#define DDT										0.003385 / 2. / 2. / 2. / 1.41421356237
#elif defined (case_1024)
	#define block_ncc					(1024*1024)
	#define block_header_size			64
 	#define my_n_part							1024UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2.
#elif defined (case_1200)
	#define block_ncc					(1200*1200)
	#define block_header_size			64
 	#define my_n_part							1200UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 1.171875
#elif defined (case_1448)
	#define block_ncc					(1448*1448)
	#define block_header_size			64
 	#define my_n_part							1448UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 1.41421356237
#elif defined (case_1800)
	#define block_ncc					(1800*1800)
	#define block_header_size			64
 	#define my_n_part							1800UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 1.41421356237 / 1.24309
#elif defined (case_1850)
	#define block_ncc					(1850*1850)
	#define block_header_size			64
 	#define my_n_part							1850UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 1.41421356237 / 1.27762
#elif defined (case_1870)
	#define block_ncc					(1870*1870)
	#define block_header_size			64
 	#define my_n_part							1870UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 1.41421356237 / 1.291436
#elif defined (case_2048)
	#define block_ncc					(2048*2048)
	#define block_header_size			64
 	#define my_n_part							2048UL
	#define DDT										0.003385 / 2. / 2. / 2. / 2. / 2.
#else
	#define block_ncc					(64*64)
	#define block_header_size			64
 	#define my_n_part							64
	#define DDT										0.003385	
#endif

// Define global constants
#define NX			my_n_part
#define NY			my_n_part
#define NZ			my_n_part

#define DX			((2. * M_PI) / ((double)NX))
#define DY			((2. * M_PI) / ((double)NY))
#define DZ			((2. * M_PI) / ((double)NZ))

#define DT 			DDT

#define GAMA		1.4
#define MINF		0.1
#define RE			1600.0
#define PR			0.71

//const int32_t NX = my_n_part;
//const int32_t NY = my_n_part;
//const int32_t NZ = my_n_part;
//const double DX = (2. * M_PI) / ((double)NX);
//const double DY = (2. * M_PI) / ((double)NY);
//const double DZ = (2. * M_PI) / ((double)NZ);
//const double DT = DDT;
//// const double DT = 1;
//const double GAMA = 1.4;
//const double MINF = 0.1;
//const double RE = 1600.0;
//const double PR = 0.71;
const double RKOLD[] = {1.0/4.0, 3.0/20.0, 3.0/5.0};
const double RKNEW[] = {2.0/3.0, 5.0/12.0, 3.0/5.0};

// constants block_* must be adapted to the data set
#define block_general_doubles		16
#define block_doubles_per_mol		9

#define c_mol_max					64
#define block_cell_list_ints		(block_ncc*c_mol_max)	// [int32_t]

#define block_i_general_dxdydz		0
#define block_i_general_nx			1
#define block_i_general_ny			2
#define block_i_general_nz			3
#define block_i_general_npart		4
#define block_i_general_ipart		5
#define block_i_general_nmol		6
#define block_i_general_nstep		7
#define block_i_general_add_U		8
#define block_i_general_add_V		9

#define block_offset_nm_cell		block_general_doubles		// [double]
#define block_offset_cell_list		(block_offset_nm_cell+(block_ncc*sizeof(int32_t))/sizeof(double))				// [double]
#define block_offset_mol			(block_offset_cell_list +(block_cell_list_ints*sizeof(int32_t))/sizeof(double))	// [double]


// no modifications required below

#define CHECK_INT					123456

#define mem_state_free			-1
#define mem_state_ready			-2
#define mem_state_ready_b		-3
#define mem_state_bussy			-4

#define mem_type_input			1
#define mem_type_output			2
#define mem_type_store			3
#define mem_type_worker			4

#define n_worker_event			64

#define n_h_store_max			4096

struct mem_info {
	int32_t type;
	int32_t i_part;
	int32_t state;
	int32_t n_use;
	int32_t i_event;
	int32_t i_cycle;
};

class DS {
	public:
		// functions
		// constructor, destructor
		DS(int32_t igpu, int32_t nworker, int32_t npart, int32_t o_in, int32_t o_out, int32_t myid, int32_t nprocs, int32_t nrails);
		~DS();
		int64_t MyGetTime();

		int32_t thread_input(int32_t n_super_cycle, int32_t myID, int32_t nProcs);
		int32_t thread_output(int32_t n_super_cycle, int32_t myID, int32_t nProcs);
		int32_t thread_input_ucx(int argc, char ** argv, int32_t n_super_cycle, int32_t myID, int32_t nProcs);
		int32_t thread_output_ucx(int argc, char ** argv, int32_t n_super_cycle, int32_t myID, int32_t nProcs);
		int32_t thread_main(int32_t n_super_cycle, int32_t myID);
		int32_t thread_storage (int32_t n_super_cycle, int32_t myID, int32_t nProcs);

		int32_t thread_store_input (int32_t n_super_cycle, int32_t myID, int32_t nProcs);
		int32_t thread_store_output (int32_t n_super_cycle, int32_t myID, int32_t nProcs);

		void InitHostStore(int32_t n);
		void FreeHostStore(int32_t n);

        void CudaDummy();

	private:
		int64_t * * pointer_list;
		int32_t n_pointer;
		int32_t store_size;
		int32_t size_device;
		int32_t size_debug;
		int32_t size_develop_a;
		int32_t n_store_in;
		int32_t n_store_worker;
		int32_t n_store_host;
		size_t size_temp;

		int32_t n_store_out;

		volatile mem_info * stat_mem_in;
		volatile mem_info * stat_mem_out;
		volatile mem_info * stat_mem_worker;
		volatile mem_info * stat_mem_store;

		CUdeviceptr * d_store;
		CUdeviceptr * d_in;
		CUdeviceptr * d_out;
		CUdeviceptr * d_worker;

		double * h_store [n_h_store_max];

		// algorithm specific data
		CUdeviceptr d_debug;

		CUdeviceptr d_visual;

		CUdeviceptr d_u0_c, d_u0_l, d_u0_r, d_u0_ll, d_u0_rr;
		CUdeviceptr d_u1_c, d_u1_l, d_u1_r, d_u1_ll, d_u1_rr;
		CUdeviceptr d_u2_c, d_u2_l, d_u2_r, d_u2_ll, d_u2_rr;
		CUdeviceptr d_p_c, d_p_l, d_p_r, d_p_ll, d_p_rr;
		CUdeviceptr d_T_c, d_T_l, d_T_r, d_T_ll, d_T_rr;

		CUdeviceptr d_du0dx, d_du0dy, d_du0dz;
		CUdeviceptr d_du1dx, d_du1dy, d_du1dz;
		CUdeviceptr d_du2dx, d_du2dy, d_du2dz;

		CUdeviceptr d_drhodx, d_drhody, d_drhodz;

		CUdeviceptr d_drhou0dx, d_drhou0dy, d_drhou0dz;
		CUdeviceptr d_drhou1dx, d_drhou1dy, d_drhou1dz;
		CUdeviceptr d_drhou2dx, d_drhou2dy, d_drhou2dz;

		CUdeviceptr d_dpdx, d_dpdy, d_dpdz;

		CUdeviceptr d_drhou0u0dx, d_drhou0u1dy, d_drhou0u2dz;
		CUdeviceptr d_drhou1u0dx, d_drhou1u1dy, d_drhou1u2dz;
		CUdeviceptr d_drhou2u0dx, d_drhou2u1dy, d_drhou2u2dz;

		CUdeviceptr d_du0d2x, d_du0d2y, d_du0d2z;
		CUdeviceptr d_du1d2x, d_du1d2y, d_du1d2z;
		CUdeviceptr d_du2d2x, d_du2d2y, d_du2d2z;

		CUdeviceptr d_du0dxdy, d_du0dxdz;
		CUdeviceptr d_du1dxdy, d_du1dydz;
		CUdeviceptr d_du2dxdz, d_du2dydz;

		CUdeviceptr d_drhoEdx, d_drhoEdy, d_drhoEdz;

		CUdeviceptr d_drhoEu0dx, d_drhoEu1dy, d_drhoEu2dz;

		CUdeviceptr d_dpu0dx, d_dpu1dy, d_dpu2dz;

		CUdeviceptr d_dTd2x, d_dTd2y, d_dTd2z;

		CUdeviceptr d_Res_rho, d_Res_rhou0, d_Res_rhou1, d_Res_rhou2, d_Res_rhoE;

		CUdeviceptr tmp_du0d2xi, tmp_du1d2xi, tmp_du2d2xi;


		//int32_t n_cycle_sample;

		//CUdeviceptr d_develop_a;

		// end of algorithm specific data

		cudaStream_t stream_in;
		cudaStream_t stream_out;
		cudaStream_t stream_worker;

		cudaEvent_t worker_event [n_worker_event];

		int32_t worker_threads_per_block;
		int32_t worker_n_block;

		int32_t i_gpu;
		int32_t n_worker;
		int32_t n_worker_total;
		int32_t n_part;
		int32_t order_in;
		int32_t order_out;

		MPI_Request request_results[2];
		bool outstanding_results;
		int32_t outstanding_results_n_cycle_sample;
		double vdata_send_a[3*1024];
		double vdata_rec_a[3*1024];
		int64_t vdata_send_b[1024];
		int64_t vdata_rec_b[1024];
		int32_t my_id;
		int32_t n_procs;
		int32_t n_rails;

		char * MyNewCatchChar(const char * file, int32_t line, int64_t size);

		// control logic
		int32_t part_in_present_wait (int32_t i_part, int32_t i_cycle);
		void part_out_ready_wait (int32_t i_part, int32_t i_center);
		void update_mem_info(int32_t *islot, volatile mem_info *mem, int32_t * i_event);
		
		// block management
		int64_t get_block_size_host(double * p_data);
		int64_t get_block_size_device(double * p_data);
        int32_t block_check(double *p_data,int32_t pos);
		int32_t block_check_device (double* data, int32_t pos);
        int32_t block_get_nm(double *p_data);
		int64_t get_block_nm_device(double * p_data);
		int32_t block_mark(double * p_data);
		int32_t block_markcheck(double * p_data);

        // I/O routines
		int32_t FileFieldsToMem(int64_t *dat, char *FileName, int32_t iMax, int64_t *size);
		int64_t MemToFile(int64_t * dat, int64_t n, char * FileName, int32_t newfile);

		// cuda routines
		void cudaCheckError(int32_t line, const char *file);
        int32_t InitCuda();
        void FreeCuda();
		void caller_worker(double **in, double **out, int32_t i_part, int32_t i_super_cycle, int32_t order_in, int32_t order_out, int32_t iworker, int32_t nworker, cudaStream_t *stream, int32_t gridSize, int32_t blockSize, int32_t myID);
		void caller_output_vtk(double * in, double * out, cudaStream_t * stream, int32_t threads_per_block, int32_t blockSize, int32_t myID, int32_t i_cycle);

		void write_vtu (float * p_data, int32_t n_mol, int32_t i_part, int32_t i_cycle);

		void caller_output_vtk_rectilinear (double * p_in, double * p_out, cudaStream_t * stream, int32_t threads_per_block, int32_t blockSize, int32_t myID, int32_t i_cycle, int32_t i_part);
		void write_vtk_rectilinear (double * p_data, int32_t n_mol, int32_t i_part, int32_t i_cycle);
		void write_vtr (double * p_data, int32_t i_part, int32_t i_cycle);

};
