// Data Streaming for Explicit Algorithms - DSEA

#include <dsea.h>
#include <iostream>
#include <cuda.h>

using namespace std;

void DS::cudaCheckError(int32_t line, const char * file) {
	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
			printf("Cuda failure %s:%d: '%s'\n", file, line,cudaGetErrorString(e));
			exit(EXIT_FAILURE);
	}
}

//int32_t DS::InitGPUMem () {
//}

void DS::CudaDummy() {
	cudaSetDevice(i_gpu);										cudaCheckError(__LINE__,__FILE__);
}

int32_t DS::InitCuda() {
	int32_t nGPU_available=-1;
	cudaGetDeviceCount(&nGPU_available);						cudaCheckError(__LINE__,__FILE__);
	cout << "nGPU_available: " << nGPU_available << endl;
	cout << "using GPU " << i_gpu << endl;

	cudaSetDevice(i_gpu);										cudaCheckError(__LINE__,__FILE__);


	if (store_size%8!=0) cout << "store_size_must_be_multiple_of_8!" << endl;
	if (size_device%8!=0) cout << "size_device_must_be_multiple_of_8!" << endl;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// host memory - pinned
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//	mem_host_size = 1024*1024*1024;
//	mem_host_size *= 8;

//cudaHostAlloc((void **)&mem_host_i64a,store_size,cudaHostAllocDefault);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// device memory
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int64_t n_alloc=0;

	d_store = new CUdeviceptr [n_worker*3];
	for (int32_t is=0;is<n_worker*3;is++) {
		cudaMalloc((void**)&d_store[is],size_device);					cudaCheckError(__LINE__,__FILE__);
		n_alloc+=size_device;
	}

	d_in = new CUdeviceptr [n_store_in];
	for (int32_t is=0;is<n_store_in;is++) {
		cudaMalloc((void**)&d_in[is],size_device);						cudaCheckError(__LINE__,__FILE__);
		n_alloc+=size_device;
	}

	d_out = new CUdeviceptr [n_store_out];
	for (int32_t is=0;is<n_store_out;is++) {
		cudaMalloc((void**)&d_out[is],size_device);						cudaCheckError(__LINE__,__FILE__);
		n_alloc+=size_device;
	}

	if (n_worker>1) {
		d_worker = new CUdeviceptr [n_store_worker*(n_worker-1)];
		for (int32_t is=0;is<n_store_worker*(n_worker-1);is++) {
			cudaMalloc((void**)&d_worker[is],size_device);					cudaCheckError(__LINE__,__FILE__);
			n_alloc+=size_device;
		}
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// algorithm specific device memory
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int32_t size_helper_array = block_ncc*my_n_part*sizeof(double);

	size_debug=1024*1024;
	cudaMalloc((void**)&d_debug,size_debug);									cudaCheckError(__LINE__,__FILE__);
	n_alloc+=size_debug;

	cudaMalloc((void**)&d_u0_c,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u0_l,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u0_r,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u0_ll,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u0_rr,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_u1_c,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u1_l,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u1_r,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u1_ll,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u1_rr,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_u2_c,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u2_l,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u2_r,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u2_ll,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_u2_rr,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_p_c,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_p_l,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_p_r,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_p_ll,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_p_rr,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_T_c,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_T_l,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_T_r,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_T_ll,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_T_rr,block_ncc*sizeof(double));			cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du0dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du0dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du1dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du1dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du2dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du2dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhodx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhody,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhodz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou0dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou0dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou1dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou1dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou2dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou2dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_dpdx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dpdy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dpdz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou0u0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou0u1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou0u2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou1u0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou1u1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou1u2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhou2u0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou2u1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhou2u2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du0d2x,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du0d2y,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du0d2z,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du1d2x,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du1d2y,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du1d2z,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du2d2x,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du2d2y,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du2d2z,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du0dxdy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du0dxdz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du1dxdy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du1dydz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_du2dxdz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_du2dydz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhoEdx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhoEdy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhoEdz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_drhoEu0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhoEu1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_drhoEu2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_dpu0dx,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dpu1dy,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dpu2dz,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_dTd2x,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dTd2y,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_dTd2z,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&d_Res_rho,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_Res_rhou0,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_Res_rhou1,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_Res_rhou2,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&d_Res_rhoE,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	cudaMalloc((void**)&tmp_du0d2xi,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&tmp_du1d2xi,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);
	cudaMalloc((void**)&tmp_du2d2xi,block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
	n_alloc+=block_ncc*sizeof(double);

	if (DOUTPUT != -1) {
		cudaMalloc((void**)&d_visual,my_n_part*block_n_fields*block_ncc*sizeof(double));		cudaCheckError(__LINE__,__FILE__);
		n_alloc+=block_ncc*c_mol_max*2*sizeof(int32_t);
	} else {
		cudaMalloc((void**)&d_visual,0);		cudaCheckError(__LINE__,__FILE__);
		n_alloc+=0;
	}



	cout << "!>M<" << (double)n_alloc/1.0e9 << "_#n_alloc" << endl;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// algorithm specific host memory
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// streams
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cudaStreamCreateWithFlags(&stream_worker,cudaStreamNonBlocking);		cudaCheckError(__LINE__,__FILE__);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// events
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	for (int i_event=0;i_event<n_worker_event;i_event++) {
		cudaEventCreate(&worker_event[i_event]);
	}

	return 0;
}

void DS::InitHostStore(int32_t n) {
	n_store_host=n;
	for (int32_t i=0;i<n_h_store_max;i++) {
		h_store[i]=(double*)-1;
	}

	for (int32_t i=0;i<n_store_host;i++) {
		cudaHostAlloc((void **)&h_store[i],store_size,cudaHostAllocDefault);
	}
}

void DS::FreeHostStore(int32_t n) {
	for (int32_t i=0;i<n_store_host;i++) {
		cudaFree((void *)h_store[i]);
	}
}

void DS::FreeCuda() {
	for (int32_t is=0;is<n_worker*3;is++) {
		cudaFree((void*)d_store[is]);				cudaCheckError(__LINE__,__FILE__);
	}
	delete [] d_store;

	for (int32_t is=0;is<n_store_in;is++) {
		cudaFree((void*)d_in[is]);					cudaCheckError(__LINE__,__FILE__);
	}
	delete [] d_in;

	for (int32_t is=0;is<n_store_out;is++) {
		cudaFree((void*)d_out[is]);					cudaCheckError(__LINE__,__FILE__);
	}
	delete [] d_out;

	for (int32_t is=0;is<n_store_worker*(n_worker-1);is++) {
		cudaFree((void*)d_worker[is]);					cudaCheckError(__LINE__,__FILE__);
	}
	delete [] d_worker;

}

char * DS::MyNewCatchChar(const char * file, int32_t line, int64_t size) {
	char * ptmp = new (std::nothrow) char [size];			// allocate new memory
	if ((!ptmp)||(ptmp==NULL)) {
		cout << "allocation of memory failed: " << file << " " << line << " " << size << endl;
		exit(1);
	}
	return ptmp;
}
