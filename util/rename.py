# 批量修改文件名
# 批量修改图片文件名
import os
import re
import sys

path = r'F:\HSI\seg\jpg'
fileList = os.listdir(path)  # 待修改文件夹


currentpath = os.getcwd()  # 得到进程当前工作目录
os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
print("start rename")
for fileName in fileList:  # 遍历文件夹中所有文件
    pat = ".+\.(jpg|png|json)"  # 匹配文件名正则表达式
    pattern = re.findall(pat, fileName)  # 进行匹配
    if pattern == []:
        continue
    name1 = fileName.split(".")
    name2 = name1[0].split("_")
    name3 = int(name2[1])
    os.rename(fileName, (name2[0]  + "_" + str(name3) + "."+pattern[0]))  # 文件重新命名

os.chdir(currentpath)  # 改回程序运行前的工作目录
sys.stdin.flush()  # 刷新
print("finish rename")