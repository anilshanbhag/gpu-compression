#pragma once
#include <cub/cub.cuh>
using namespace cub;

__forceinline__ __device__ int decodeElementDBin(int i, uint* data_block) {
  // Reference for the frame
  int reference = reinterpret_cast<int*>(data_block)[0];

  // Miniblock bitwidth
  uint bitwidth = data_block[1] & 255;

  uint start_bitindex = (bitwidth * i);
  uint start_intindex = 2 + (start_bitindex >> 5);
  start_bitindex = start_bitindex & (32-1);

  /*uint* data_block_uint = reinterpret_cast<uint*>(data_block);*/
  unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) | data_block[start_intindex];
  uint element = (element_block & (((1LL<<bitwidth) - 1LL) << start_bitindex)) >> start_bitindex;

  return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadDBinPack(uint* block_start, 
    uint* data, uint* shared_buffer, int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;

  // Specialize BlockScan for a 1D block of 128 threads on type int
  typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

  int tile_idx = blockIdx.x;

  // Block start indices of 5 blocks converted into integer offsets.
  uint *block_starts = &shared_buffer[0];
  if (threadIdx.x < 5) {
    block_starts[threadIdx.x] = block_start[tile_idx * 4 + threadIdx.x];
  }
  __syncthreads();

  // Shared memory for 4 blocks of encoded l_shipdate data 
  uint* data_block = &shared_buffer[5];

  // Lets load 4 blocks from the encoded column
  uint start_offset = block_starts[0] - 1;
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7); // i*128
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
  __syncthreads();

  int first_value = data_block[0];
  data_block = data_block + 1;

  for (int i=0; i<4; i++) {
    if (is_last_tile) {
      if (threadIdx.x + i*BLOCK_THREADS < num_tile_items) {
        items[i] = decodeElementDBin(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
      }
    }
    else {
      items[i] = decodeElementDBin(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
    }
  }

  if (threadIdx.x == 0) {
    items[0] = first_value;
  }

  __syncthreads();

  typename BlockScan::TempStorage *temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
  typename BlockExchange::TempStorage *temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

  BlockExchange(*temp_storage_exchange).StripedToBlocked(items);

  __syncthreads();

  /*// Also accepts an initial value.*/
  BlockScan(*temp_storage_scan).InclusiveSum(items, items);

  __syncthreads();
  BlockExchange(*temp_storage_exchange).BlockedToStriped(items);
}

