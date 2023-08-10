#pragma once
#include <cub/cub.cuh>
using namespace cub;

__forceinline__ __device__ int decodeElementRBin(int i, uint* data_block, uint reference, uint bitwidth) {

  uint start_bitindex = (bitwidth * i);
  uint start_intindex = (start_bitindex >> 5); // 3 for reference bitwidth and count

  start_bitindex = start_bitindex & (32-1);

  /*unsigned long long element_block = *((unsigned long long*)&data_block[miniblock_offset + start_intindex]);*/
  unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) | data_block[start_intindex];
  uint element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

  return reference + element;
}


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack(uint* val_block_start, uint* rl_block_start,
    uint* value, uint* run_length, uint* shared_buffer, int (&items_value)[ITEMS_PER_THREAD], 
    int (&items_run_length)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

  typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;

  // Specialize BlockScan for a 1D block of 128 threads on type int
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

  uint num_decode;

  int tile_idx = blockIdx.x;

  // Block start indices of 5 blocks converted into integer offsets.
  uint *val_block_starts = &shared_buffer[0];
  uint *rl_block_starts = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3) + BLOCK_THREADS * (ITEMS_PER_THREAD)];
  if (threadIdx.x  < 2) {
    val_block_starts[threadIdx.x] = val_block_start[tile_idx + threadIdx.x];
    rl_block_starts[threadIdx.x] = rl_block_start[tile_idx + threadIdx.x];
  }

  __syncthreads();

  // Shared memory for 4 blocks of encoded l_shipdate data 
  uint* val_data_block = &val_block_starts[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];
  uint* rl_data_block = &rl_block_starts[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  // // Lets load 4 blocks from the encoded column
  uint start_offset_val = val_block_starts[0];
  uint end_offset_val = val_block_starts[1];
  uint start_offset_rl = rl_block_starts[0];
  uint end_offset_rl = rl_block_starts[1];

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset_val + threadIdx.x + (i * BLOCK_THREADS); // i * 128
    if (index < end_offset_val) {
      val_data_block[threadIdx.x + (i * BLOCK_THREADS)] = value[index];
    }
    index = start_offset_rl + threadIdx.x + (i * BLOCK_THREADS); // i * 128
    if (index < end_offset_rl) {
      rl_data_block[threadIdx.x + (i * BLOCK_THREADS)] = run_length[index];
    }
  }

  __syncthreads();

  uint count = val_data_block[2]; // == rl_ptr[2]
  uint offset = 0;
  num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {

    uint* val_ptr = val_data_block + 3;
    uint* rl_ptr = rl_data_block + 3;

    uint reference, bitwidth;
    if (threadIdx.x < num_decode) {
      reference = val_data_block[0];
      bitwidth = val_data_block[1] & 255;
      items_value[i] = decodeElementRBin(threadIdx.x + offset, val_ptr, reference, bitwidth);
      reference = rl_data_block[0];
      bitwidth = rl_data_block[1] & 255;
      items_run_length[i] = decodeElementRBin(threadIdx.x + offset, rl_ptr, reference, bitwidth);
    } else {
      items_value[i] = 0;
      items_run_length[i] = 0;
    }

    offset += num_decode;

  }

  __syncthreads();
  
  typename BlockScan::TempStorage *temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(rl_data_block);
  typename BlockExchange::TempStorage *temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(rl_data_block);

  BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);

  __syncthreads();

  /*// Also accepts an initial value.*/
  BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    val_data_block[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
  }

  __syncthreads();

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    val_data_block[items_run_length[i]] = 1;
  }

  __syncthreads();

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    items_run_length[i] = val_data_block[threadIdx.x * ITEMS_PER_THREAD + i];
  } 

  __syncthreads();

  BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);

  __syncthreads();

  BlockExchange(*temp_storage_exchange).BlockedToStriped(items_run_length);

  __syncthreads();


  offset = 0;
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (threadIdx.x < num_decode) val_data_block[threadIdx.x + offset] = items_value[i];
    offset += num_decode;
  }

  __syncthreads();


  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    items_value[i] = val_data_block[items_run_length[i]];
  }
}