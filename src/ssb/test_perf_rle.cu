#define CUB_STDERR

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "../kernel.cuh"
#include "utils/gpu_utils.h"
#include "ssb_gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator  g_allocator(true);


template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRBinKernel(
    int* col, 
    uint* val_block_start, uint* val_data, uint* rl_block_start, uint* rl_data,
    int num_entries) {
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;
  int tile_offset = tile_idx * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int val_block[ITEMS_PER_THREAD];
  int rl_block[ITEMS_PER_THREAD];

  int num_tiles = (num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_entries - tile_offset;
    is_last_tile = true;
  }

  extern __shared__ uint shared_buffer[];
  LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(val_block_start, rl_block_start,
    val_data, rl_data, shared_buffer, val_block, rl_block, is_last_tile, num_tile_items);

  __syncthreads();

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    col[tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x] = val_block[i];
  }
}


float runSinglePass(
    encoded_column val_col, encoded_column rl_col,
    int num_items, string encoding,
    CachingDeviceAllocator&  g_allocator, int* col) {
  // Kernel timing
  float time_query;
  SETUP_TIMING();

  int *unpack_bitpack = NULL, *for_decoded = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &unpack_bitpack, num_items * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**) &for_decoded, num_items * sizeof(int)));

  const int num_threads = 128;
  const int items_per_thread = 4; //the only difference for RLE
  int tile_size = num_threads * items_per_thread;

  // Run kernel
  cudaEventRecord(start, 0);
  if (encoding == "rbin") {
    TIME_FUNC((runRBinKernel<num_threads, items_per_thread><<<(num_items + tile_size - 1)/tile_size, num_threads, 4096>>>(
      col, 
      val_col.block_start, val_col.data, rl_col.block_start, rl_col.data,
      num_items 
    )), time_query);

  } 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  return time_query;
}

/***
  * The goal is to test is col encoding can be decoded and if it same as original array.
  */
int main(int argc, char** argv) {
  int num_trials = 3;

  if (argc != 2) return 0;

  //./bin/ssb/test_match_rle lo_orderkey
  string column_name = argv[1];
  string encoding = "rbin";

  int len = LO_LEN;

  encoded_column val_col = loadEncodedColumnToGPURLE(column_name, "valbin", len, g_allocator);
  encoded_column rl_col = loadEncodedColumnToGPURLE(column_name, "rlbin", len, g_allocator);

  int *col;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &col, len * sizeof(int)));

  cudaDeviceSynchronize();

  for (int t = 0; t < num_trials; t++) {
    float time_query;

    time_query = runSinglePass(val_col, rl_col,
                     len, encoding,
                     g_allocator, col);

    cout << "{" << "\"query\":6" << ",\"time_query\":" << time_query << "}" << endl;

    cudaDeviceSynchronize(); 
  }

  return 0;
}
