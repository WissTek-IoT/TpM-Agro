import os

file_location  = os.path.join(os.path.dirname(__file__), '../L4_Storage/data_to_predict.txt')

data_file = open(file_location, 'w+')
data_file.write("1\n")
data_file.write("a\n")
data_file.write("b\n")
data_file.write("c\n")
data_file.close()

a=[1,2,3,4]
print(a[:-1])