#!/bin/python

import sys

ocena_test_rec = sys.argv[1]
ocena_studenta = sys.argv[2] 


f_test = open(sys.argv[1],'r')
f_student = open(sys.argv[2],'r')

test = dict()
student = dict()

for line in f_test:
    list_split =  line.split()
    test[list_split[0]] = list_split[1]
f_test.close()


for line in f_student:
    list_split =  line.split()
    student[list_split[0]] = list_split[1]
f_student.close()

good = 0

for key in test:
    if test[key] == student[key]:
        good += 1
ratio = float(good)/600.0
print(ratio) 
if ratio < 0.75:
    exit(1)
