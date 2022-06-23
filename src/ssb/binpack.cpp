#include "ssb_utils.h"
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>

using namespace std;

int delta(int*& in, int*& out, int num_entries) {
  for (int i = num_entries - 1; i > 0; i--) {
    out[i] = in[i] - in[i-1];
  }

  out[0] = 0;
  return 0;
}

uint binPack(uint*&in, uint*& out, uint*& block_offsets, uint num_entries) {
  uint offset = 0;

  uint block_size = 128;
  uint miniblock_count = 4;
  uint total_count = num_entries;
  uint first_val = in[0];

  out[0] = block_size;
  out[1] = miniblock_count;
  out[2] = total_count;
  out[3] = first_val;

  offset += 4;

  uint miniblock_size = uint(block_size / miniblock_count);
  uint num_blocks = (num_entries + block_size - 1) / block_size;

  for (uint block_start=0; block_start<num_entries; block_start += block_size) {
    uint block_index = block_start / block_size;
    block_offsets[block_index] = offset;

    // Find min val
    uint min_val = in[0];
    for (int i = 1; i < block_size; i++) {
      if (in[i] < min_val) min_val = in[i];
    }

    for (int i = 0; i < block_size; i++) {
      in[i] = in[i] - min_val;
    }

    uint miniblock_size = block_size / miniblock_count;
    uint* miniblock_bitwidths = new uint[miniblock_count];
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = 0;

    for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
      for (uint i = 0; i < miniblock_size; i++) {
        uint bitwidth = uint(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
        if (bitwidth > miniblock_bitwidths[miniblock]) miniblock_bitwidths[miniblock] = bitwidth;
      }
    }

    // Extra for Simple BinPack
    uint max_bitwidth = miniblock_bitwidths[0];
    for (int i=1; i<miniblock_count; i++) max_bitwidth = max(max_bitwidth, miniblock_bitwidths[i]);
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = max_bitwidth;
    if (block_start == 0) cout << "max_bitwidth " << max_bitwidth << endl;

    out[offset] = min_val;
    offset++;

    out[offset] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) +
      (miniblock_bitwidths[2] << 16) + (miniblock_bitwidths[3] << 24);
    offset++;

    for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
      uint bitwidth = miniblock_bitwidths[miniblock];
      uint shift = 0;
      for (int i = 0; i < miniblock_size; i++) {
        if (shift + bitwidth > 32) {
          if (shift != 32) out[offset] += in[miniblock * miniblock_size + i] << shift;
          offset++;
          shift = (shift + bitwidth) & (32-1);
          out[offset] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
        } else {
          out[offset] += in[miniblock * miniblock_size + i] << shift;
          shift += bitwidth;
        }
      }
      offset++;
    }

    // Increment the input pointer by block size
    in += block_size;
  }

  block_offsets[num_entries / block_size] = offset;

  return offset;
}

int storeEncodedColumn(string col_name, uint* col, uint arr_byte_size, uint* offsets, uint num_blocks) {
  string filename = DATA_DIR + lookup(col_name) + ".bin";
  ofstream colData (filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    cout << "Unable to write column" << endl;
    return -1;
  }

  colData.write((char*)col, arr_byte_size);
  colData.close();

  string offsets_filename = DATA_DIR + lookup(col_name) + ".binoff";
  ofstream offsetsData (offsets_filename.c_str(), ios::out | ios::binary);
  if (!offsetsData) {
    cout << "Unable to write offsets" << endl;
    return -1;
  }

  offsetsData.write((char*)offsets, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "encode <col-name>" << endl;
    return 1;
  }

  string col_name = argv[1];
  int len = LO_LEN;

  uint *raw = loadColumn<uint>(col_name, len);
  cout << "Loaded Column" << endl;

  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint *col = new uint[adjusted_len];
  memcpy(col, raw, len * sizeof(int));

  uint *out = new uint[adjusted_len]();
  uint *offsets = new uint[num_blocks + 1]();

  // extend with the last value to make it multiple of 128
  for (int i = len; i < adjusted_len ;i++) col[i] = raw[len-1];

  uint arr_byte_size = binPack(col, out, offsets, adjusted_len);
  cout << "Num Elements " << len << endl;
  cout << "Input: ArrSize " << len * 4 << endl;
  cout << "Output: ArrSize " << arr_byte_size << " Offsets " << num_blocks + 1 << endl;

  storeEncodedColumn(col_name, out, arr_byte_size * 4, offsets, num_blocks);

  return 0;
}

