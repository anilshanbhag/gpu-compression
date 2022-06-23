#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "../ssb/ssb_utils.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "./bin/bench/gen_d3 <num_bits>" << endl;
    return 0;
  }

  int alpha = atoi(argv[1]);
  cout << "Encoding with " << alpha << " alpha" << endl;
  string col_name = "testd3_" + to_string(alpha);
  int len = 1<<28;

  ifstream ifs;
  ifs.open("/home/ubuntu/deltafor/test/bench/zipf/datad3_" + to_string(alpha), std::ifstream::in);

  int gl = 1<<20;
  int* arr = new int[gl];

  for (int i = 0; i < gl; i++) {
    ifs >> arr[i];
  }

  for (int i=0; i<10; i++)
    cout << arr[i] << " ";
  cout << endl;

  ifs.close();

  uint* raw = new uint[len];

  for (int i=0; i<len; i += gl) {
    for (int j=i; j<i+gl; j++) {
      raw[j] = arr[j-i];
    }
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

