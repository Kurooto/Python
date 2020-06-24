import numpy as np
import matplotlib.pyplot as plt
import glob
import pathlib
import re
import pandas as pd
path='data_sample/jikken1'


class Data:
	def __init__(self):
		self.elec=[0]*2

	def setNumber(self,num):
		self.num=num

	def setWeight(self,weight):
		self.weight=weight

	def setTime(self,time):
		self.time=time

	def setElec_1(self,elec_1):
		self.elec[0]=elec_1

	def setElec_2(self,elec_2):
		self.elec[1]=elec_2


class Dataset:
	def __init__(self):
		self.weight=[]
		self.elec=[0]*2
		self.elec[0]=[]
		self.elec[1]=[]

	def Add(self,weight,elec_1,elec_2):
		self.weight.append(weight)
		self.elec[0].append(elec_1)
		self.elec[1].append(elec_2)


def separate_data(file_name):
	f=open(file_name)
	data=f.read().split()
	time=[]
	elec_1=[]
	elec_2=[]
	for i in range(int(len(data)/3)):
		time.append(float(data[3*i]))
		elec_1.append(float(data[3*i+1]))
		elec_2.append(float(data[3*i+2]))
	f.close()
	return time,elec_1,elec_2


def make_list(path):
	n=len(glob.glob("./"+path+"/*"))
	p_temp = pathlib.Path(path).glob('*.txt')
	list=[]
	for p in p_temp:
		data=Data()
		data.setNumber(int(p.name[15]))
		weight=re.sub("\\D", "", p.name)
		data.setWeight(int(weight[2:]))
		file_name=path+"/"+p.name
		time,elec_1,elec_2=separate_data(file_name)
		data.setTime(time)
		data.setElec_1(elec_1)
		data.setElec_2(elec_2)
		list.append(data)
	return list


def sort_data(list1,list2,list3):
	for i in range(len(list1)-1):
		max_value=max(list1[i:])
		max_index=list1.index(max_value)
		x1=list1[i]
		x2=list2[i]
		x3=list3[i]
		list1[i]=list1[max_index]
		list2[i]=list2[max_index]
		list3[i]=list3[max_index]
		list1[max_index]=x1
		list2[max_index]=x2
		list3[max_index]=x3
	return list1,list2,list3


def make_graph(list,n):
	dataset=[0]*3
	for i in range(3):
		dataset[i]=Dataset()

	for i in range(len(list)):
		ave_1=(sum(np.square(list[i].elec[0]))/len(list[i].elec[0]))**0.5
		ave_2=(sum(np.square(list[i].elec[1]))/len(list[i].elec[1]))**0.5
		k=list[i].num-1
		dataset[k].Add(list[i].weight,ave_1,ave_2)

	for i in range(3):
		dataset[i].weight,dataset[i].elec[0],dataset[i].elec[1]=sort_data(dataset[i].weight,dataset[i].elec[0],dataset[i].elec[1])
		plt.plot(dataset[i].weight,dataset[i].elec[n-1],marker='.',label="trainee"+str(i+1)+"(elec_"+str(n)+')')
		s1=pd.Series(dataset[i].weight)
		s2=pd.Series(dataset[i].elec[n-1])
		res=s1.corr(s2)
		print(i+1,res)
	plt.grid()
	plt.xlabel("Weight(g)")
	plt.ylabel("RMS(V)")
	plt.title("Relationship between Weight and RMS")
	plt.legend()
	plt.savefig("jikken1-"+str(n))
	plt.show()
	return




list=make_list(path)
make_graph(list,1)
make_graph(list,2)
