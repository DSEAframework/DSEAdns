// Data Streaming for Explicit Algorithms - DSEA

#include "dsea.h"
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

// This routine determines the actual size of a block in bytes. The maximum size is that of the storage, i.e. store_size bytes
int64_t DS::get_block_size_host(double * p_data) {
	int64_t block_size=block_header_size*sizeof(double)+block_n_fields*block_ncc*sizeof(double);
	return block_size;
}

// This routine determines the block size of a block stored in GPU memory
int64_t DS::get_block_size_device(double * p_data) {
	int64_t block_size=block_header_size*sizeof(double)+block_n_fields*block_ncc*sizeof(double);
	return block_size;
}

// This routine determines the block size of a block stored in GPU memory
int64_t DS::get_block_nm_device(double * p_data) {
	int64_t i_tmp=-1;
	// cudaMemcpy((void*)&i_tmp,(const void*)&p_data[block_i_general_nmol],sizeof(int64_t),cudaMemcpyDeviceToHost);		cudaCheckError(__LINE__,__FILE__);
	return i_tmp;
}

// This routine checks the block for consistency
int32_t DS::block_check(double * p_data, int32_t pos) {
	return 0;
}

int32_t DS::block_check_device (double* data, int32_t pos) {
	return 0;
}


// This routine checks the block for consistency
int32_t DS::block_get_nm(double * p_data) {
	return 0;
}

int32_t DS::block_mark(double * p_data) {
	return 0;
}

int32_t DS::block_markcheck(double * p_data) {
	return 0;
}
