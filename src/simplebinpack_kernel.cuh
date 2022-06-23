#pragma once

__forceinline__ __device__ int decodeElement(int i, uint* data_block) {
  // Reference for the frame
  uint reference = data_block[0];
  uint bitwidth = data_block[1] & 255;
  uint start_bitindex = (bitwidth * i);
  uint start_intindex = 2 + (start_bitindex >> 5);

  start_bitindex = start_bitindex & (32-1);

  unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) | data_block[start_intindex];
  uint element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

  return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack(uint* block_start, 
    uint* data, uint* shared_buffer, int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int tile_idx = blockIdx.x;
  int threadId = threadIdx.x;

  // Block start indices of 5 blocks converted into integer offsets.
  uint *block_starts = &shared_buffer[0];
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
  __syncthreads();

  // Shared memory for 4 blocks of encoded l_shipdate data 
  // 5 + 32
  uint* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  // Lets load 4 blocks from the encoded column
  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7); // i * 128
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
  __syncthreads();

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
      items[i] = decodeElement(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
  }
}

