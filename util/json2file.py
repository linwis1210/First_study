import os
import labelme

path = "./test1/"  # path是你存放json的路径
all_file = os.listdir(path)
json_file = []
for file in all_file:
    if file.endswith('.json'):
        json_file.append(file)
del all_file
for file in json_file:
    os.system("python C:/Users/JLB/anaconda3/Scripts/labelme_json_to_dataset.exe %s"%(path + file))