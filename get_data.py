import numpy as np
import pandas as pd
import math
import csv
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy

def get_data():
	
	N_SAMPLES = 101766
	N_DRUGS = 23
	N_NUMERICAL = 8
	N_CATEGORICAL = 3

	label_readmission = np.zeros(N_SAMPLES,dtype = int)
	readmission_Dict = {
	'NO':0,
	'<30':1,
	'>30':2
	}

	label_readmission_two_class = np.zeros(N_SAMPLES, dtype = int)
	readimission_Two_Class_Dict = {
	'NO':0,
	'<30':1,
	'>30':1
	}

	label_HBA1C = np.zeros(N_SAMPLES,dtype = int)
	HBA1C_Dict = {
	'>8':0,
	'>7':1,
	'Norm':2,
	'None':3
	}

	bagOfDrugs_Dict ={
	'Down':1,
	'Up':1,
	'Steady':1,
	'No':0
	}

	age_dict = {
	0:1,
	1:1,
	2:1,
	3:1,
	4:1,
	5:2,
	6:3,
	7:4,
	8:5,
	9:5
	}

	label_diag1 = np.zeros(N_SAMPLES,dtype = int)
	data_categorical = np.zeros(shape = (N_SAMPLES, N_CATEGORICAL), dtype= np.float64)
	data_numerical = np.zeros(shape = (N_SAMPLES, N_NUMERICAL), dtype= np.float64)
	data_bagOfDrugs = np.zeros(shape = (N_SAMPLES, N_DRUGS), dtype = int)



	with open('diabetic_data.csv') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		i = 0
		# skip the first iteration
		iterrows = iter(reader)
		next(iterrows)
		for row in iterrows:
			data_categorical[i][0:N_CATEGORICAL] = [int(num) for num in row[6:6+N_CATEGORICAL]]

			#parse Age, age is in column 4
			age = int(row[4][1:row[4].find('-')])
			data_numerical[i][0] = age_dict[age/10]
			
			#parse Time in the hospital, which is column 9
			data_numerical[i][1] = int (row[9])
			#parse the rest of the numerical data, which is in column 12:18
			data_numerical[i][2:] = [int(num) for num in row [12:18]]

			#parse bag of drugs, which is in column 24:47
			drug_dosage = row[24:47]
			for j in range (0, len(drug_dosage)):
				data_bagOfDrugs[i][j] = bagOfDrugs_Dict[drug_dosage[j]]
			
			
			#parse the readimission label, which is last column
			label_readmission[i] = readmission_Dict[row[-1]]
			label_readmission_two_class[i] = readimission_Two_Class_Dict[row[-1]]

			#parse the HBA1C test label, which is column 23
			label_HBA1C[i] = HBA1C_Dict[row[23]]
			#parse the Primary Diagonosis, which is column 18
			if(row[18].find('?') != -1):
				label_diag1[i] = 9
			elif(row[18].find('V') !=-1):
				label_diag1[i] = 9
			elif(row[18].find('E')!=-1):
				label_diag1[i] = 9
			else:

				diag1 = int(float(row[18]))

				if(( diag1>= 390 and diag1<=459) or diag1 == 785):
					label_diag1[i] = 1
				elif(diag1 == 250):
					label_diag1[i] = 2
				elif((diag1 >= 468 and diag1 <=519) or diag1 == 786):
					label_diag1[i] = 3
				elif((diag1 >= 520 and diag1 <=579) or diag1 == 787):
					label_diag1[i] = 4
				elif(diag1 >= 800 and diag1 <=999):
					label_diag1[i] = 5
				elif(diag1 >= 710 and diag1 <=739):
					label_diag1[i] = 6
				elif((diag1 >= 580 and diag1 <=629) or diag1 == 788):
					label_diag1[i] = 7
				elif(diag1 >= 140 and diag1 <=239):
					label_diag1[i] = 8
				else:
					label_diag1[i] = 9
			label_diag1[i]-=1
			i+= 1



	x_data = deepcopy(data_bagOfDrugs)
	for i in range(len(data_categorical[0])):
		x_data = np.concatenate((x_data,pd.get_dummies(data_categorical[:,i]).values) ,axis = 1)
	x_data = np.concatenate((x_data , data_numerical) , axis = 1)

	
	y_data = [ label_readmission_two_class, label_HBA1C, label_diag1, label_readmission]
	y_data_labels = ["readmission", "HBA1C", "diag1", "readmission_3classes"]

	return x_data,y_data,y_data_labels


