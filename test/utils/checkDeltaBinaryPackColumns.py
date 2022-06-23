import struct
import sys

if __name__ == "__main__":
  check_file = open("data/" + sys.argv[1] + ".dat", "rb")
  data = check_file.read()

  print "File has", len(data), "bytes"

  block_size = struct.unpack('<i', data[0:4])[0]
  print "Block size:", block_size

  miniblock_count = struct.unpack('<i', data[4:8])[0]
  print "Miniblock count:", miniblock_count

  total_count = struct.unpack('<i', data[8:12])[0]
  print "Total count:", total_count

  print "First value:",  struct.unpack('<i', data[12:16])[0]

  num_deltas = total_count - 1
  rem = (total_count-1) % block_size
  if rem > 0:
    rem = block_size - rem
    num_deltas += rem

  block_count = num_deltas / block_size
  miniblock_size = block_size / miniblock_count

  curr_byte = 16
  first = True
  for i in range(block_count):
    print "curr byte", curr_byte

    min_delta = struct.unpack('<i', data[curr_byte:curr_byte+4])[0]
    print "Min delta of block", i, ":", min_delta
    curr_byte += 4

    bitwidths = struct.unpack('<iiii', data[curr_byte:curr_byte+16])
    print "Bitwidths of miniblocks in block", i, ":", bitwidths
    curr_byte += 16

    for j in range(len(bitwidths)):
      if first:
        arr = []
        for i in range((bitwidths[j] * miniblock_size)/32):
           arr += [struct.unpack('<i', data[curr_byte+i*4:curr_byte+i*4+4])[0]]
        print arr
        first = False

      total_bitwidth = (bitwidths[j] * miniblock_size)
      curr_byte += (total_bitwidth / 8)
