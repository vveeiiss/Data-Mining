import numpy as np
import pandas as pd


df_train = pd.read_csv("/Users/melikadavoodzade/Downloads/VU_DM_data/training_set_VU_DM.csv")
df_test = pd.read_csv("/Users/melikadavoodzade/Downloads/VU_DM_data/test_set_VU_DM.csv")
df_new = df_train[['orig_destination_distance',
 'prop_location_score2',
 'prop_review_score',
 'click_bool',
 'srch_id',
 'random_bool',
 'srch_room_count',
 'site_id',
 'visitor_location_country_id',
 'prop_country_id',
 'prop_id',
 'prop_starrating',
 'prop_brand_bool',
 'prop_location_score1',
 'prop_log_historical_price',
 'position',
 'price_usd',
 'promotion_flag',
 'srch_destination_id',
 'srch_length_of_stay',
 'srch_booking_window',
 'srch_adults_count',
 'srch_children_count',
 'srch_saturday_night_bool',
 'booking_bool']]
label_train = df_train["booking_bool"]


def labeling(train):
	pass

def get_data():
	x_train = df_new.to_numpy()
	y_train = label_train.to_numpy()
	return x_train, y_train

	