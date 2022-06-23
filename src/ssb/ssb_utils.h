#pragma once

#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>
#include <string>

/*#include <cuda.h>*/
/*#include <cub/util_allocator.cuh>*/

using namespace std;
//using namespace cub;

#define SF 20

#define BASE_PATH "/home/ubuntu/deltafor/test/ssb/data/"

#if SF == 1
#define DATA_DIR BASE_PATH "s1_columnar/"
#define LO_LEN 6001171
#define P_LEN 200000
#define S_LEN 2000
#define C_LEN 30000
#define D_LEN 2556
#elif SF == 10
#define DATA_DIR BASE_PATH "s10_columnar/"
#define LO_LEN 59986214
#define P_LEN 800000
#define S_LEN 20000
#define C_LEN 300000
#define D_LEN 2556
#else // 20
#define DATA_DIR BASE_PATH "s20_columnar/"
#define LO_LEN 119994368
//#define LO_LEN 119994746
#define P_LEN 1000000
#define S_LEN 40000
#define C_LEN 600000
#define D_LEN 2556
#endif

int index_of(string* arr, int len, string val) {
  for (int i=0; i<len; i++)
    if (arr[i] == val)
      return i;

  return -1;
}

// 16 / 6 / 7 / 8 - not integer columns
string lookup(string col_name) {
  string lineorder[] = { "lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate", "lo_orderpriority", "lo_shippriority", "lo_quantity", "lo_extendedprice", "lo_ordtotalprice", "lo_discount", "lo_revenue", "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"};
  string part[] = {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container"};
  string supplier[] = {"s_suppkey", "s_name", "s_address", "s_city", "s_nation", "s_region", "s_phone"};
  string customer[] = {"c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment"};
  string date[] = {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year", "d_yearmonthnum", "d_yearmonth", "d_daynuminweek", "d_daynuminmonth", "d_daynuminyear", "d_sellingseason", "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"};

  if (col_name[0] == 'l') {
    int index = index_of(lineorder, 17, col_name);
    return "LINEORDER" + to_string(index);
  } else if (col_name[0] == 's') {
    int index = index_of(supplier, 7, col_name);
    return "SUPPLIER" + to_string(index);
  } else if (col_name[0] == 'c') {
    int index = index_of(customer, 8, col_name);
    return "CUSTOMER" + to_string(index);
  } else if (col_name[0] == 'p') {
    int index = index_of(part, 9, col_name);
    return "PART" + to_string(index);
  } else if (col_name[0] == 'd') {
    int index = index_of(date, 15, col_name);
    return "DDATE" + to_string(index);
  } else if (col_name[0] == 't') {
    // test columns
    return "../../../bench/data/" + col_name;
  } else {
    cout << "Unknown column " << col_name << endl;
    exit(1);
  }

  return "";
}

template<typename T>
T* loadColumn(string col_name, int num_entries) {
  T* h_col = new T[num_entries];
  string filename = DATA_DIR + lookup(col_name);
  ifstream colData (filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    return NULL;
  }

  colData.read((char*)h_col, num_entries * sizeof(T));
  return h_col;
}

template<typename T>
int storeColumn(string col_name, int num_entries, T* h_col) {
  string filename = DATA_DIR + lookup(col_name);
  ofstream colData (filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    return -1;
  }

  colData.write((char*)h_col, num_entries * sizeof(T));
  return 0;
}

struct encoded_column {
  // block_start[i] = byte at which ith block starts
  uint* block_start;
  // raw data
  uint* data;
  // number of bytes of raw data
  int data_size;
};

/***
 * Loads encoding from disk into memory
 * encoding: bin | dbin
 **/
encoded_column loadEncodedColumn(string col_name, string encoding, int num_entries) {
  if (!(encoding == "bin" || encoding == "dbin" || encoding == "pbin")) {
    cout << "Encoding has to be bin or dbin" << endl;
    exit(1);
  }

  // Open file
  string filename = DATA_DIR + lookup(col_name) + "." + encoding;
  string offsets_filename = DATA_DIR + lookup(col_name) + "." + encoding + "off";

  int fd = open(filename.c_str(), O_RDONLY);

  // Get size of file
  struct stat s;
  int status = fstat(fd, &s);
  int filesize = s.st_size;

  encoded_column col;

  ifstream colData (filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    cout << "Unable to open encoded column file" << filename << endl;
    exit(1);
  }

  col.data = new uint[filesize / 4];
  colData.read((char*)col.data, filesize);
  colData.close();

  col.data_size = filesize;

  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((num_entries + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  col.block_start = new uint[num_blocks + 1];

  ifstream offsetsData (offsets_filename.c_str(), ios::in | ios::binary);
  if (!offsetsData) {
    cout << "Unable to open encoded column file" << offsets_filename << endl;
    exit(1);
  }

  offsetsData.read((char*)col.block_start, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return col;
}

/*int main() {*/
  //int *h_col = new int[10];
  //for (int i=0; i<10; i++) h_col[i] = i;
  //storeColumn<int>("test", 10, h_col);
  //int *l_col = loadColumn<int>("test", 10);
  //for (int i=0; i<10; i++) cout << l_col[i] << " ";
  //cout << endl;
  //return 0;
/*}*/


