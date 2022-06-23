import sys
import csv

if __name__ == "__main__":
  # Get locations file
  location_file_name = sys.argv[1] 
  location_file = open(location_file_name, "r")
  locations = location_file.read().splitlines()

  # Get name of table to split into columns
  table_name = sys.argv[2]

  # Get table file name
  table_dir = locations[0]
  table_file_name = table_dir + "/" + table_name + ".csv"

  # Read in table
  table_file = open(table_file_name, "r")
  table_file_lines = table_file.read().splitlines() 

  # Get number of columns in table
  num_columns = len(table_file_lines[0].split("|"))

  # Divide table into columns
  table_data = []
  for _ in range(num_columns):
    table_data.append([])

  for table_file_line in table_file_lines:
    split_line = table_file_line.split("|")
    for i in range(num_columns):
      table_data[i].append(split_line[i])

  print("NUMBER OF ELEMENTS IN TABLE IS " + str(len(table_data[0])))

  # Get schema file 
  schema_file_name = "schemas/" + table_name + "_schema.txt"
  schema_file = open(schema_file_name, "r") 
  schema = schema_file.read().splitlines()

  # Get column names 
  column_names = []
  for line in schema:
    column_names.append(line.split(",")[0])

  # Get names of 1-char columns
  symbol_column_names = ["l_returnflag", "l_linestatus"]

  # Country/region columns
  nation_dict = {
    "ALGERIA": 0,
    "ARGENTINA": 1, 
    "BRAZIL": 2, 
    "CANADA": 3, 
    "EGYPT": 4, 
    "ETHIOPIA": 5, 
    "FRANCE": 6,
    "GERMANY": 7, 
    "INDIA": 8,
    "INDONESIA": 9,
    "IRAN": 10, 
    "IRAQ": 11, 
    "JAPAN": 12,
    "JORDAN": 13,
    "KENYA": 14, 
    "MOROCCO": 15, 
    "MOZAMBIQUE": 16,
    "PERU": 17,
    "CHINA": 18,
    "ROMANIA": 19,
    "SAUDI ARABIA": 20, 
    "VIETNAM": 21,
    "RUSSIA": 22,
    "UNITED KINGDOM": 23, 
    "UNITED STATES": 24
  }

  region_dict = {
    "AFRICA": 0,
    "AMERICA": 1, 
    "ASIA": 2, 
    "EUROPE": 3,
    "MIDDLE EAST": 4 
  }

  mktsegment_dict = {
    "AUTOMOBILE": 0,
    "BUILDING": 1, 
    "FURNITURE": 2, 
    "MACHINERY": 3, 
    "HOUSEHOLD": 4
  }

  # Save each column to data/TABLE/COLUMN_NAME.csv
  for i in range(num_columns):
    column_file_name = "data/" + table_name + "/" + column_names[i] + ".txt" 
    column_file = open(column_file_name, "w+")
    if "date" in column_names[i]:
      for data in table_data[i]:
        fixed_data = data.replace("-", "") # remove hyphens from dates 
        column_file.write(fixed_data + " ")
    elif column_names[i] in symbol_column_names:
      for data in table_data[i]:
        fixed_data = str(ord(data.lower()[0])) # convert text to int
        column_file.write(fixed_data + " ")
    elif column_names[i] == "n_name":
      for data in table_data[i]:
        fixed_data = str(nation_dict[data])
        column_file.write(fixed_data + " ")
    elif column_names[i] == "r_name":
      for data in table_data[i]:
        fixed_data = str(region_dict[data])
        column_file.write(fixed_data + " ")
    elif column_names[i] == "c_mktsegment":
      for data in table_data[i]:
        fixed_data = str(mktsegment_dict[data])
        column_file.write(fixed_data + " ")
    else:
      for data in table_data[i]:
        column_file.write(data + " ") 
    column_file.close()
