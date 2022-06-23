// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <cmath>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

#include "cub/test/test_util.h"
#include "utils/gpu_utils.h"

#include "ssb_utils.h"

using namespace std;
using namespace cub;

int main() {
  string cols[] = {
    "orderkey",
    "linenumber",
    "custkey",
    "partkey",
    "suppkey",
    "orderdate",
    "quantity",
    "extendedprice",
    "ordtotalprice",
    "discount",
    "revenue",
    "supplycost",
    "tax",
    "commitdate"
  };

  int num_cols = 14;

  int maxval;
  for (int i=0; i<num_cols; i++) {
    int *col = loadColumn<int>("lo_" + cols[i], LO_LEN);
    maxval = 0;
    for (int j=0; j<LO_LEN; j++) {
      if (col[j] == 119994746 || col[j] == 60548584 || col[j] == 15137146) continue;
      if (col[j] > maxval) {
        maxval = col[j];
      }
    }

    int bitwidth = uint(ceil(log2(maxval + 1)));
    /*cout << cols[i] << " " << bitwidth << endl;*/
    if (bitwidth > 16) cout << LO_LEN * 4 << endl;
    else if (bitwidth > 8) cout << LO_LEN * 2 << endl;
    else cout << LO_LEN << endl;
  }

  return 0;
}




