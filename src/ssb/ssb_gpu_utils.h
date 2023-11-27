#pragma once

#include "ssb_utils.h"
#include "cub/test/test_util.h"
using namespace cub;

template<typename T>
T* loadColumnToGPU(T* src, int len, CachingDeviceAllocator& g_allocator) {
  T* dest = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &dest, sizeof(T) * len));
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
  return dest;
}

encoded_column loadEncodedColumnToGPU(string col_name, string encoding, int len, CachingDeviceAllocator& g_allocator) {
  if (!(encoding == "bin" || encoding == "dbin" || encoding == "pbin")) {
    cout << "Encoding has to be bin or dbin" << endl;
    exit(1);
  }

  encoded_column h_col = loadEncodedColumn(col_name, encoding, len);

  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1, g_allocator);
  uint* d_col_data = loadColumnToGPU<uint>(h_col.data, h_col.data_size/4, g_allocator);

  cout << "Encoded Col Size: " << h_col.data_size << " " << num_blocks + 1 << endl;

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  return d_col;
}

encoded_column loadEncodedColumnToGPURLE(string col_name, string encoding, int len, CachingDeviceAllocator& g_allocator) {
  if (!(encoding == "valbin" || encoding == "rlbin")) {
    cout << "Encoding has to be valbin or rlbin" << endl;
    exit(1);
  }

  encoded_column h_col = loadEncodedColumnRLE(col_name, encoding, len);

  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1, g_allocator);
  uint* d_col_data = loadColumnToGPU<uint>(h_col.data, h_col.data_size/4, g_allocator);

  cout << "Encoded Col Size: " << h_col.data_size << " " << num_blocks + 1 << " " << h_col.data_size + num_blocks + 1 << endl;

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  return d_col;
}
