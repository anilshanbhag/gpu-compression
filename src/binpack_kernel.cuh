#pragma once

__forceinline__ __device__ int decodeElement(int i, uint miniblock_index, uint index_into_miniblock, uint* data_block, uint* bitwidths, uint* offsets) {
  // Reference for the frame
  int reference = reinterpret_cast<int*>(data_block)[0];

  uint miniblock_offset = offsets[miniblock_index];
  uint bitwidth = bitwidths[miniblock_index];

  uint start_bitindex = (bitwidth * index_into_miniblock);
  uint start_intindex = 2 + (start_bitindex >> 5);

  start_bitindex = start_bitindex & (32-1);

  unsigned long long element_block = (((unsigned long long)data_block[miniblock_offset + start_intindex + 1]) << 32) | data_block[miniblock_offset + start_intindex];
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

  uint* bitwidths = &shared_buffer[ITEMS_PER_THREAD + 1];
  uint* offsets = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    // Miniblock bitwidths
    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);

    // Miniblock bitwidth
    uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;
  }
  __syncthreads();

  // Index of miniblock containing i
  uint miniblock_index = threadIdx.x >> 5; // i / 32

  // Entry index in the miniblock
  uint index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    /*if (is_last_tile) {*/
      /*if (threadIdx.x + i*128 < num_tile_items) {*/
        /*items[i] = decodeElement(threadIdx.x, data_block + block_starts[i] - block_starts[0]);*/
      /*}*/
    /*}*/
    /*else {*/
      items[i] = decodeElement(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
    /*}*/
  }
}

