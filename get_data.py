import numpy as np
import pandas as pd


df_train = pd.read_csv("/Users/melikadavoodzade/Downloads/VU_DM_data/training_set_VU_DM.csv")
df_test = pd.read_csv("/Users/melikadavoodzade/Downloads/VU_DM_data/test_set_VU_DM.csv")
df_new = df_train.iloc[:, 0:52]
label_train = df_train["booking_bool"]


def labeling(train):
	pass

def get_data():
	x_train = df_new.to_numpy()
	y_train = label_train.to_numpy()
	return x_train, y_train

	