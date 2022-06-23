import struct
import binascii
import math

if __name__ == "__main__":
  # Names of int32/int64 columns
  column_names = [ "lineitem/l_orderkey",
                   "lineitem/l_shipdate",
                   "lineitem/l_quantity",
                   "orders/o_orderdate",
                   "orders/o_shippriority",
                   "orders/o_custkey",
                   "orders/o_orderkey",
                   "customer/c_custkey",
                   "customer/c_mktsegment"]
  column_names = [ "lineitem/l_partkey", ]
  first = True

  for i in range(len(column_names)):
    # New output file of binary data
    fout = open("data/" + column_names[i] + ".dat", 'w+b')

    # Read in un-encoded column
    column_name = "data/" + column_names[i] + ".txt"
    print("Encoding " + column_name)
    before_column_file = open(column_name, "r")
    before_column_data = before_column_file.read().split()
    before_column_data = [int(x) for x in before_column_data]

    # Block count size
    block_size = 128
    fout.write(struct.pack('<i', block_size))

    # Number of miniblocks per block
    miniblock_count = 4
    fout.write(struct.pack('<i', miniblock_count))

    # Total count
    total_count = len(before_column_data)
    fout.write(struct.pack('<i', total_count))

    # First value
    first_val = before_column_data[0]
    fout.write(struct.pack('<i', first_val))

    # Pad last block with 0's if last block is not full
    num_deltas = total_count
    deltas = before_column_data
    print("Initial Data")
    print(deltas[:128])
    rem = (total_count) % block_size
    if rem > 0:
      rem = block_size - rem
      deltas.extend([0] * rem)
      num_deltas += rem

    print("First Value", first_val)

    # Minimum deltas for each block
    min_deltas = []

    # Compute deltas
    j = 0
    min_delta = deltas[j] - first_val
    for k in range(num_deltas):
      # Compute delta
      deltas[k] -= first_val

      # Compute min delta of block
      min_delta = min(min_delta, deltas[k])
      if (j+1) % block_size == 0:
        min_deltas.append(min_delta)
        if j+1 < num_deltas:
          min_delta = deltas[j+1] - first_val

      j += 1
    min_deltas.append(min_delta)
    print("Min delta", min_deltas[0])

    print("Deltas")
    print(deltas[:128])

    # Make deltas > 0 by subtracting min delta
    for k in range(num_deltas):
      deltas[k] -= min_deltas[int(k/block_size)]

    print("Final Block")
    print(deltas[:128])

    # Size of miniblock
    miniblock_size = int(block_size / miniblock_count)

    # Find max bits of each miniblock (list of lists of miniblock_count elements)
    mini_bitwidths = []
    k = 0
    while k < num_deltas:
      # List of max bits for this block
      mini_bitwidth = []

      for j in range(miniblock_count):
        # Min(x) s.t. 2^x >= all elements in current miniblock
        max_delta = max(deltas[k:k+miniblock_size])
        if max_delta == 0:
          max_delta = 1
        bitwidth = int(math.ceil(math.log(max_delta, 2.0)))+1
        mini_bitwidth.append(bitwidth)

        k += miniblock_size

      mini_bitwidths.append(mini_bitwidth)

    # Write [min_delta, list of mini_bitwidths, list of miniblocks] for each block
    block_count = int(num_deltas / block_size)
    for k in range(block_count):
      # Write min_delta of current block
      fout.write(struct.pack('<i', min_deltas[k]))
      # print "Min delta of block", k, min_deltas[k]

      # Write list of mini_bitwidths for current block
      for j in range(miniblock_count):
        fout.write(struct.pack('<i', mini_bitwidths[k][j]))

      # Binary pack miniblocks
      for j in range(miniblock_count):
        miniblock_str = ""
        fmt = "{0:0" + str(mini_bitwidths[k][j]) + "b}"
        for l in range(miniblock_size):
          # print "Formatting ", deltas[k*block_size + j*miniblock_size+l], ":", fmt.format(deltas[k*block_size + j*miniblock_size+l])
          # fmt has a weird behaviour
          # If we have array [11, 7] each represented as 4 bit entries
          # We want 11010111
          # fmt reverses representation when printing - so we get 11 as 1011
          # and simply combining gives 10111110
          # [::-1] reverses the array to get the right behaviour
          en_str = fmt.format(deltas[k*block_size + j*miniblock_size+l])[::-1]
          if first:
              print(en_str)
          miniblock_str += en_str
        if first:
            print(miniblock_str)
            first = False
        # print "Block", k, "Miniblock", j, miniblock_str

        # Write binary packed miniblocks
        l = 0
        while l < len(miniblock_str):
          # Write 1 byte at a time
          # We again reverse string to parse it correctly
          chunk = int(miniblock_str[l:l+8][::-1], 2)
          # print "Writing out ", miniblock_str[l:l+8], "as", chunk, "or", bytearray([chunk])
          fout.write(bytearray([chunk]))
          l += 8

    fout.close()
