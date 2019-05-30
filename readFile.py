import csv
# 读取csv至字典
csvFile = open("data_t/edge.csv", "r")
reader = csv.reader(csvFile)

# 建立空字典
result = {}
for item in reader:
    print(item)

csvFile.close()
