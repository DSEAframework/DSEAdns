// Data Streaming for Explicit Algorithms - DSEA

#include "dsea.h"
#include <fstream>

using namespace std;

int64_t DS::MemToFile(int64_t * dat, int64_t size, char * FileName, int32_t newfile) {
	ofstream ofs; 
	if (newfile==1) {
		ofs.open(FileName, ios::out | ios::binary);
		// cout << FileName << " new " << n << endl;
	}
	else if (newfile==0) {
		ofs.open(FileName, ios::out | ios::binary | ios::app);
	}
	else {
		cout << "invalid MemToFile" << endl;
		return -1;
	}

	if (ofs) {
		ofs.write((char*)dat,size);
		if (ofs) {
			ofs.close();
			return size;
		}
		else {
			cout << "problem writing file: " << FileName << " " << size << endl;
			cout << "aborting..." << endl;
		}
		ofs.close();
	}
	else {
		cout << "problem opening file: " << FileName << endl;
	}
	return -1;
}


int32_t DS::thread_output (int32_t n_super_cycle, int32_t myID, int32_t nProcs) {
	int32_t i_store=0;

	int64_t n_mol_stored=0;



	for (int32_t i_supercycle=0;i_supercycle<n_super_cycle;i_supercycle++) {
		if ((i_supercycle==n_super_cycle-1)&&(myID==nProcs-1)) {
			// last cycle in last MPI rank stores output
			char * my_block = MyNewCatchChar(__FILE__,__LINE__,store_size);

			for (int32_t i_part=0;i_part<n_part;i_part++) {
				// cout << "OUT:start!_" << i_store << "_" << stat_mem_out[i_store].i_event << endl;
				while (stat_mem_out[i_store].i_event==-1) {}	// wait for event

				// cout << "OUT:wait!_" << i_store << "_" << stat_mem_out[i_store].i_event << endl;
				cudaError_t ces=cudaEventSynchronize (worker_event[stat_mem_out[i_store].i_event]);

				if (ces==cudaSuccess) {
					// download block from GPU
					cudaMemcpy((void*)my_block,(const void*)d_out[i_store],store_size,cudaMemcpyDeviceToHost);		cudaCheckError(__LINE__,__FILE__);
					// block_check((double*)my_block,2);
					// block_markcheck((double*)my_block);

					// cout << "OUT:ready!_" << i_store << endl;
					stat_mem_out[i_store].i_event=-1;
					stat_mem_out[i_store].i_part=-1;
					stat_mem_out[i_store].n_use=0;
					stat_mem_out[i_store].state=mem_state_free;

					i_store++;
					if (i_store==n_store_out) i_store=0;
				}
			}

			delete [] my_block;

		}
		else {
			// regular cycle
			for (int32_t i_part=0;i_part<n_part;i_part++) {
				while (stat_mem_out[i_store].i_event==-1) {}	// wait for event

				cudaError_t ces=cudaEventSynchronize (worker_event[stat_mem_out[i_store].i_event]);
				if (ces==cudaSuccess) {
					int32_t dest=myID+1;
					if (dest==nProcs) dest=0;
					int32_t tag=i_part;

					int32_t send_size=get_block_size_device((double*)d_out[i_store]);

					int32_t res=MPI_Send((const void*)d_out[i_store],send_size/sizeof(int32_t),MPI_INT,dest,tag,MPI_COMM_WORLD);

					if (res==MPI_SUCCESS) {
						stat_mem_out[i_store].i_event=-1;
						stat_mem_out[i_store].i_part=-1;
						stat_mem_out[i_store].n_use=0;
						stat_mem_out[i_store].state=mem_state_free;
					}

					i_store++;
					if (i_store==n_store_out) i_store=0;
				}
			}
		}
	}

	return 0;
}
