#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include "../ssb/ssb_utils.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "./bin/bench/gen_d1 <num_bits>" << endl;
    return 0;
  }

  int num_distinct = atoi(argv[1]);
  cout << "Encoding with " << (1 << num_distinct) << " values" << endl;
  string col_name = "testd1_" + to_string(num_distinct);
  int len = 1<<28;

  uint* raw = new uint[len];
  int segment_len = len / (1<<num_distinct);

  for (int i=0; i<len; i += segment_len) {
    int val = (i / segment_len) + 1;
    for (int j=i; j<i+segment_len; j++) {
      raw[i] = val;
    }
  }

  cout << "Generated Column" << endl;

  cout << "Writing to " << DATA_DIR + lookup(col_name) << endl;

  storeColumn<uint>(col_name, len, raw);

  cout << "Stored Column" << endl;

  return 0;
}

