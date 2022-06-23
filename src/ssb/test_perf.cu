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
#include "ssb_gpu_utils.h"
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator  g_allocator(true);

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runBinKernel(
    int* col, 
    uint* col_block_start, uint* col_data,
    int num_entries) {
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;
  int tile_offset = tile_idx * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int col_block[ITEMS_PER_THREAD];

  int num_tiles = (num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_entries - tile_offset;
    is_last_tile = true;
  }

  extern __shared__ uint shared_buffer[];
  LoadBinPack(col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();

  for (int i=0; i<4; i++) {
    col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel(
    int* col, 
    uint* col_block_start, uint* col_data,
    int num_entries) {
  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;
  int tile_offset = tile_idx * tile_size;

  // Load a segment of consecutive items that are blocked across threads
  int col_block[ITEMS_PER_THREAD];

  int num_tiles = (num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_entries - tile_offset;
    is_last_tile = true;
  }

  extern __shared__ uint shared_buffer[];
  LoadDBinPack(col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();

  for (int i=0; i<4; i++) {
    col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
  }
}

float runTest(
    encoded_column d_col,
    int num_items, string encoding,
    CachingDeviceAllocator&  g_allocator, int* col) {
  // Kernel timing
  float time_query;
  SETUP_TIMING();

  int tile_size = 512;

  // Run kernel
  if (encoding == "bin") {
    TIME_FUNC((runBinKernel<128,4><<<(num_items + tile_size - 1)/tile_size, 128, 3000>>>(
      col, 
      d_col.block_start, d_col.data, 
      num_items 
    )), time_query);
  } else if (encoding == "dbin") {
    TIME_FUNC((runDBinKernel<128,4><<<(num_items + tile_size - 1)/tile_size, 128, 3000>>>(
      col, 
      d_col.block_start, d_col.data, 
      num_items 
    )), time_query);
  }

  return time_query;
}

/***
  * The goal is to test is col encoding can be decoded and if it same as original array.
  */
int main(int argc, char** argv) {
  int num_trials = 1; 
  string column_name = "lo_extendedprice";
  string encoding = "bin";

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);
  args.GetCmdLineArgument("c", column_name);
  args.GetCmdLineArgument("e", encoding);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--t=<num trials>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  encoded_column d_col = loadEncodedColumnToGPU(column_name, encoding, LO_LEN, g_allocator);

  int *d_out;
  ALLOCATE(d_out, LO_LEN * sizeof(int));

  cudaDeviceSynchronize();

  // Run trials
  for (int t = 0; t < num_trials; t++) {
    float time_query = runTest(d_col,
                               LO_LEN, encoding,
                               g_allocator, d_out);

    cout << "{" << "\"query\":6" << ",\"time_query\":" << time_query << " ms}" << endl;

    cudaDeviceSynchronize(); 
  }

  return 0;
}

