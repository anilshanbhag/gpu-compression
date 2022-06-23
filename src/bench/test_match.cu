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
  LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
  }
}

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
  LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

  __syncthreads();

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    col[tile_size * tile_idx + i * 128 + threadIdx.x] = col_block[i];
  }
}

float runTest(
    encoded_column d_col,
    int num_items, string encoding,
    CachingDeviceAllocator&  g_allocator, int* h_col_orig) {
  // Kernel timing
  float time_query;
  SETUP_TIMING();

  int* col = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &col, num_items * sizeof(int)));

  const int num_threads = 128;
  const int items_per_thread = 4;
  int tile_size = num_threads * items_per_thread;

  // Run kernel
  if (encoding == "bin") {
    TIME_FUNC((runBinKernel<num_threads, items_per_thread><<<(num_items + tile_size - 1)/tile_size, num_threads, 3000>>>(
      col, 
      d_col.block_start, d_col.data, 
      num_items 
    )), time_query);
  } else if (encoding == "dbin") {
    TIME_FUNC((runDBinKernel<num_threads, items_per_thread><<<(num_items + tile_size - 1)/tile_size, num_threads, 4000>>>(
      col, 
      d_col.block_start, d_col.data, 
      num_items 
    )), time_query);
  }

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Copy revenue from device to host 
  int* h_col = new int[num_items];
  CubDebugExit(cudaMemcpy(h_col, col, sizeof(int) * num_items, cudaMemcpyDeviceToHost));

  for (int i=0; i<512; i++) cout << h_col[i] << ",";
  cout << endl;

  for (int i=0; i<num_items; i++) {
    if (h_col_orig[i] != h_col[i]) {
      cout << "ERROR:" << i << " " << h_col_orig[i] << " " << h_col[i] << endl;
      return -1;
    }
  }
  cout << "Inputs match ! " << endl;

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
          "[--c=<column_name>] "
          "[--e=<encoding:bin|dbin>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int len = 1<<28;
  int *h_col_orig = loadColumn<int>(column_name, len);

#if 0
  int* raw = (int*) h_col_orig;
  int len = len;
  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  int *col = new int[adjusted_len];
  memcpy(col, h_col_orig, len * sizeof(int));

  // extend with the last value to make it multiple of 128
  for (int i = len; i < adjusted_len ;i++) col[i] = raw[len-1];

  for (uint tile_start=0; tile_start<adjusted_len; tile_start += tile_size) {
    int* in = &col[tile_start];
    // Compute the deltas
    for (int i = tile_size - 1; i > 0; i--) {
      in[i] = in[i] - in[i-1];
      /*in[i] = in[i] - in[0];*/
    }
    in[0] = 0;
  }

  h_col_orig = (int*) col;
#endif

  cout << "Encoding,Column: " << encoding << "," << column_name << endl;

  encoded_column d_col = loadEncodedColumnToGPU(column_name, encoding, len, g_allocator);

/*  int pb = 0;*/
  /*for (int i=pb*128; i<pb*128+128; i++) cout << h_col_orig[i] << endl;*/

  /*cout << "__________" << endl;*/

  /*for (int i=pb; i<pb + 1; i++) cout << h_col.block_start[i] << endl;*/

  /*cout << "__________" << endl;*/

  /*for (int i=0; i<40; i++) cout << bitset<32>(h_col.data[h_col.block_start[pb] + i]) << endl;*/

  /*cout << "__________" << endl;*/

  cudaDeviceSynchronize();

  // Run trials
  for (int t = 0; t < num_trials; t++) {
    float time_query = runTest(d_col,
                               len, encoding,
                               g_allocator, h_col_orig);

    cout << "{" << "\"query\":6" << ",\"time_query\":" << time_query << " ms}" << endl;
    cudaDeviceSynchronize(); 
  }


  return 0;
}

