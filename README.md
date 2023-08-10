# Tile-based Lightweight Integer Compression in GPU

A key constraint of GPU-based data analytics today is the limited memory capacity in GPU devices. Data compression is a powerful technique that can mitigate the capacity limitation in two ways:

* Fitting more data into GPU memory  
* Speeding up data transfer between CPU and GPU.

This package implements three bit-packing-based optimized data compression formats and their decompression routines for GPUs: GPU-FOR, GPU-DFOR, and GPU-RFOR. The work was presented at SIGMOD '22. Please read the [paper](https://dl.acm.org/doi/abs/10.1145/3514221.3526132) for more details. 

```
@inproceedings{gpubitpacking,
  author = {Shanbhag, Anil and Yogatama, Bobbi W. and Yu, Xiangyao and Madden, Samuel},
  title = {Tile-Based Lightweight Integer Compression in GPU},
  year = {2022},
  isbn = {9781450392495},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3514221.3526132},
  doi = {10.1145/3514221.3526132},
  booktitle = {Proceedings of the 2022 International Conference on Management of Data},
  pages = {1390–1403},
  numpages = {14},
  keywords = {GPU data analytics, GPU data compression, bit-packing},
  location = {Philadelphia, PA, USA},
  series = {SIGMOD '22}
}
```

Usage
---

The decompression routines are implemented as device functions. Use the routine `LoadBinPack` / `LoadDBinPack` / `LoadRBinPack` in place of a `BlockLoad` routine and point it to the appropriate compressed column. As these are device functions, you can directly use them in your own program too.

**To generate the test distributions:**

* For uniform distribution, distribution d1 and d2

```
make bench/gen bench/gen_d1 bench/gen_d2
./bin/bench/gen <num_bits>
./bin/bench/gen_d1 <num_bits>
./bin/bench/gen_d2 <num_bits>
```

* For `d3`, run the bench/gen_d3.py file

Note these will written out the DATA_DIR defined in `ssb/ssb_utils.h` as flat files.

**To generate Star Schema Benchmark data:**

Follow the instructions [here](https://github.com/anilshanbhag/crystal)

**To encode the data to GPU-\* format**

The above two steps will generate flat files which contain 4-byte integer arrays. To generate the encoded columns:

```
# For test distributions
make bench/binpack
make bench/deltabinpack
make bench/rlebinpack

./bin/bench/binpack <num_bits>
./bin/bench/deltabinpack <num_bits>
./bin/bench/rlebinpack <num_bits>

# For SSB columns
make ssb/binpack
make ssb/deltabinpack
make ssb/rlebinpack

./bin/ssb/binpack <col_name>
./bin/ssb/deltabinpack <col_name>
./bin/ssb/rlebinpack <col_name>
```

You can find test SSB implementations [here](https://github.com/anilshanbhag/crystal/tree/master/src/ssb)
Replace the `BlockLoad` routine with `LoadBinPack` / `LoadDBinPack` / `LoadRBinPack`.
