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
    cout << "./bin/bench/gen_d2 <num_bits>" << endl;
    return 0;
  }

  int mean = atoi(argv[1]);
  cout << "Encoding with " << (1 << mean) << " mean" << endl;
  string col_name = "testd2_" + to_string(mean);
  int len = 1<<28;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean, 20.0);

  uint* raw = new uint[len];

  for (int i=0; i<len; i += 1) {
    double number = distribution(generator);
    if (number < 1) number = 1;
    raw[i] = (uint) number;
  }

  cout << "Generated Column" << endl;

  cout << "Writing to " << DATA_DIR + lookup(col_name) << endl;

  storeColumn<uint>(col_name, len, raw);

  cout << "Stored Column" << endl;

  return 0;
}

