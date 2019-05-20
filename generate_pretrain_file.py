import numpy as np
import csv
import os
import random
import sys

data_dir = "data/Amazon_few_shot"
filter_name = "workspace.filtered.list"
output_file = "data/Amazon_corpus.txt"
task_list = []
with open(os.path.join(data_dir, filter_name), 'r') as f:
    task_list = f.readlines()
    task_list = [name.strip() for name in task_list]
out_file = open(output_file, 'w')
task_class = ['t2', 't5', 't4']
for task_name in task_list:
    for task in task_class:
        file_name = os.path.join(data_dir, task_name)+'.'+task+'.train'
        print(file_name)
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            
            for line in reader:
                out_file.write(line[0])
                try:
                    a = line[1]
                except:
                    print("file_name:", file_name)
                    print(line)

        out_file.write('\n\n')
out_file.close()