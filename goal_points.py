#!/usr/bin/python
#!/usr/bin/env python
import re
import numpy as np
import os
from os.path import expanduser
home = expanduser("~")
#np.set_printoptions(threshold='nan')

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
def open_yaml(filename):
   f = open(filename, 'r')
   txt=f.readlines()
   f.close()#read the yaml file line by line
   print(txt)
   a= txt[1][12:len(txt[1])-1]
   x=float(txt[2][9:19])
   y=float(txt[2][21:31])
   print(x,',',y) #the left bottlem  point of the map
   a=float(a)
   print(a)#the resolution
   return a,x,y
#transfer pgm pixel to yaml map coordinate
def trans_pgm_to_yaml(a=[] , re=float):
    x_pixel = a[0] * re + x
    y_pixel = (-y) - (a[1] * re)
    return x_pixel,y_pixel
# if this point can access,then the result should be true
def accessibility(a=[]):
    print(accessable[a[0], a[1]])
if __name__ == "__main__":

    image = read_pgm(home + "/catkin_ws/src/lab4/maps/examplemap.pgm", byteorder='<') # the map should be in the same
    yaml = home +"/catkin_ws/src/lab4/maps/examplemap.yaml"
    accessable=image>250#find the white part

    [resolution,x,y] = open_yaml(yaml)
    print(image.shape)  # matrix data of pgm image
    print(accessable)
    [rows, cols] = image.shape #the number of rows and columes

    acc_list=[]#the list of points which can be access in the map in each colume
    lists=[] # the list of points which we can access
    flag = False
    for i in range(cols):
        for j in range(rows):
            if (accessable[j][i]== True):
                corr_pgm=[int(i),int(j)]
                corr = trans_pgm_to_yaml(corr_pgm,resolution)
                acc_list.append(corr)
                flag=True
        if (flag == True):
            lists.append(acc_list)
        acc_list=[]
        flag=False
    print(len(lists))

    lable_list=[]#should be deleted
    for i in range(len(lists)):
        if (len(lists[i])<50):
            lable_list.append(i)
        print(len(lists[i]))
    for i in range(len(lable_list)):
        del lists[lable_list[i]-i]



    print('number of columes now :'+str(len(lists)))
    print('those columes should be deleted')
    print(lable_list)
    print(len(lable_list))
    list_len=int(len(lists)/8)
    print(list_len)
    count=0
    final_list= []
    for i in range(len(lists)):
        count=count+1
        if (count >= list_len):
            if(len(lists[i])>100):
                final_list.append(lists[i][int(len(lists[i]) / 5)])
                final_list.append(lists[i][int(len(lists[i]) / 2)])
                final_list.append(lists[i][int(len(lists[i]) * 4 / 5)])
            elif (len(lists[i])>70):
               final_list.append(lists[i][int(len(lists[i])/4)])
               final_list.append(lists[i][int(len(lists[i])*3/4)])
            elif (len(lists[i])>40):
                final_list.append(lists[i][int(len(lists[i]) / 2)])
            count=0

    print('the point list is')
    print(final_list)
    print('the list has '+str(len(final_list))+' points')

    corr_file=open(home +"/catkin_ws/src/lab5/points.csv", "w+") #save the points into points.txt file
    for i in final_list:
        corr_file.writelines(str(i))
        corr_file.writelines('\n')
    corr_file.close()
