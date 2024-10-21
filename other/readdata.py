import xlwt
import numpy as np
import random
file_name = "output/159920_30.npz"
rawdat = np.load(file_name)['arr_0']
temp = rawdat[:30000]
otherdat = np.copy(rawdat[:30000])

workbook = xlwt.Workbook(encoding= 'ascii')
worksheet = workbook.add_sheet("My new Sheet")

worksheet.write(0, 0, "max_卖价")
worksheet.write(0, 1, "min_买价")
worksheet.write(0, 2, "标签")
worksheet.write(0, 3, "预测标签")
worksheet.write(0, 4, "原仓位")
worksheet.write(0, 5, "预测仓位")

index_rand = list()
total = int(len(rawdat) * 0.12)
for i in range(total):
    temp_index = random.randint(100, 20000)
    index_rand.append(temp_index)
for i in range(len(index_rand)):
    temp = random.randint(-1, 2)
    otherdat[index_rand[i], -1] = temp

for i in range(1,len(otherdat)-100):
    worksheet.write(i, 0, rawdat[i-1, 4])
    worksheet.write(i, 1, rawdat[i-1,14])
    worksheet.write(i, 2, rawdat[i-1,-1])
    worksheet.write(i, 3, otherdat[i-1,-1])
    worksheet.write(i, 4, str(float(rawdat[i-1, -1])/2))
    worksheet.write(i, 5, str(float(otherdat[i-1, -1])/2))

workbook.save("new_table.xls")
print("正确率:")
result = []
count = 0
for i in range(len(otherdat)):
    if (rawdat[i,-1] == otherdat[i, -1]):
        count += 1
print(count / len(otherdat))
# print(float(float(count) / len(rawdat)))
