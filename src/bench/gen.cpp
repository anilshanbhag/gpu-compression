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
    cout << "./bin/bench/gen <num_bits>" << endl;
    return 0;
  }

  int num_bits = atoi(argv[1]);
  cout << "Encoding with " << num_bits << " bits" << endl;
  string col_name = "test" + to_string(num_bits);
  int len = 1<<28;

  uint* raw = new uint[len];
  uint mask = (1 << num_bits) - 1;

  for (int i=0; i<len; i++) {
    raw[i] = rand() & mask;
  }

  for (int i=0; i<10; i++)
    cout << raw[i] << " ";
  cout << endl;

  cout << "Generated Column" << endl;

  cout << "Writing to " << DATA_DIR + lookup(col_name) << endl;

  storeColumn<uint>(col_name, len, raw);

  cout << "Stored Column" << endl;

  return 0;
}

