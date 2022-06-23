#!/usr/bin/env bash

lines=(`cat "locations.txt"`)

if [ $1 == "help" ]; then 
  printf "Usage: ./pipeline.sh [COMMAND] [ARGS]

Commands:
gent SF
\tgenerates tpch data with a scale factor of SF
col TABLE
\tsaves each column of TABLE in data/TABLE/column.csv
query [QUERY]
\truns QUERY file
dbenc
\tdelta binary encode
prof [KERNEL] [QUERY]
\tnvprofs KERNEL of QUERY (run query command first)
dbcheck COLUMN
\tcheck delta binary encoding
\te.g. COLUMN=lineitem/l_orderkey
clean
\twipes the data directory clean\n"
elif [ $1 == "gent" ]; then
  (cd ${lines[0]}; make; ./dbgen -s $2; sed 's/.$//' lineitem.tbl > lineitem.csv; sed 's/.$//' customer.tbl > customer.csv; sed 's/.$//' orders.tbl > orders.csv; sed 's/.$//' supplier.tbl > supplier.csv; sed 's/.$//' nation.tbl > nation.csv; sed 's/.$//' region.tbl > region.csv; sed 's/.$//' part.tbl > part.csv;)
elif [ $1 == "col" ]; then
  mkdir "data/$2"; python storeColumns.py locations.txt $2
elif [ $1 == "query" ]; then
  nvcc -O3 -std=c++11 -gencode arch=compute_52,code=sm_52 $2.cu -o $2.out; ./$2.out
elif [ $1 == "prof" ]; then
  nvprof --kernels $2 --analysis-metrics -f -o analysis.prof ./$3.out
elif [ $1 == "dbenc" ]; then
  python deltaBinaryPackColumns.py 
elif [ $1 == "dbcheck" ]; then
  python checkDeltaBinaryPackColumns.py $2
elif [ $1 == "clean" ]; then
  rm -rf data; mkdir data; rm *.out
else
  printf "Command not valid. If stuck, please use help flag.\n"
fi