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
#include "../ssb/ssb_gpu_utils.h"
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

CachingDeviceAllocator  g_allocator(true);

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runBinKernel(
    unsigned long long* global_sum, 
    uint* col_block_start, uint* col_data,
    int num_entries) {
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceInt;

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    uint shared_buffer[BLOCK_THREADS * ITEMS_PER_THREAD];
    typename BlockReduceInt::TempStorage reduce;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int col_block[ITEMS_PER_THREAD];

  int num_tiles = (num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_entries - tile_offset;
    is_last_tile = true;
  }

  /*extern __shared__ uint shared_buffer[];*/
  LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, temp_storage.shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();
  unsigned long long sum = 0;
  for (int i=0; i<ITEMS_PER_THREAD; i++) sum += col_block[i]; 
  /*col_block[0] + col_block[1] + col_block[2] + col_block[3];*/
  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, aggregate);
  }


/*  for (int i=0; i<4; i++) {*/
    /*col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];*/
  /*}*/
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel(
    unsigned long long* global_sum, 
    uint* col_block_start, uint* col_data,
    int num_entries) {
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceInt;

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    uint shared_buffer[BLOCK_THREADS * ITEMS_PER_THREAD];
    typename BlockReduceInt::TempStorage reduce;
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int col_block[ITEMS_PER_THREAD];

  int num_tiles = (num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = num_entries - tile_offset;
    is_last_tile = true;
  }

  /*extern __shared__ uint shared_buffer[];*/
  LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, temp_storage.shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();

  unsigned long long sum = 0;
  for (int i=0; i<ITEMS_PER_THREAD; i++) sum += col_block[i]; 
  /*col_block[0] + col_block[1] + col_block[2] + col_block[3];*/
  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, aggregate);
  }
}

float runTest(
    encoded_column d_col,
    int num_items, string encoding,
    CachingDeviceAllocator&  g_allocator, unsigned long long* d_sum) {
  // Kernel timing
  float time_query;
  SETUP_TIMING();

  const int num_threads = 128;
  const int items_per_thread = 4;
  int tile_size = num_threads * items_per_thread;

  // Run kernel
  if (encoding == "bin") {
    TIME_FUNC((runBinKernel<num_threads, items_per_thread><<<(num_items + tile_size - 1)/tile_size, num_threads, 3000>>>(
      d_sum, 
      d_col.block_start, d_col.data, 
      num_items 
    )), time_query);
  } else if (encoding == "dbin") {
    TIME_FUNC((runDBinKernel<num_threads, items_per_thread><<<(num_items + tile_size - 1)/tile_size, 128, 3000>>>(
      d_sum,
      d_col.block_start, d_col.data,
      num_items 
    )), time_query);
  }

  unsigned long long revenue;
  CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  cout << revenue << endl;

  return time_query;
}

/***
  * The goal is to test is col encoding can be decoded and if it same as original array.
  */
int main(int argc, char** argv) {
  int num_trials = 1; 
  string column_name = "test";
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

  int len = 1<<28;

  // Initialize device
  CubDebugExit(args.DeviceInit());

  encoded_column d_col = loadEncodedColumnToGPU(column_name, encoding, len, g_allocator);

  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));

  cudaMemset(d_sum, 0, sizeof(long long));

  cudaDeviceSynchronize();

  // Run trials
  for (int t = 0; t < num_trials; t++) {
    float time_query = runTest(d_col,
                               len, encoding,
                               g_allocator, d_sum);

    cout << "{" << "\"query\":\"test\"" << ",\"time_query\":" << time_query << " ms}" << endl;

    cudaDeviceSynchronize(); 
  }

  return 0;
}

