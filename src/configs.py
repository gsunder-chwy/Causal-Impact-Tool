import os
import logging

# paths
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
SQL_PATH = '../sql'

# vertica production database connection info
VERTICA_PROD_INFO = {
    'host': 'bidb.chewy.local', 
    'port': 5433, 
    'database': 'bidb', 
    'user': os.getenv('vertica_user'), # ass username
    'password': os.getenv('vertica_password'), # add password
    'read_timeout': 3600, 
    'connection_timeout': 300, 
    'log_level': logging.ERROR
}

# postgress database connection info
POSTGRESS_PROD_INFO = {
    'host': 'chewy-dev-use1-ors-simulations-db-cluster.cluster-cbpioeretmip.us-east-1.rds.amazonaws.com', 
    'port': 5432, 
    'database': 'orssimsdb',
    #'env': 'simtool', 
    'user': os.getenv('postgres_user'), # add username
    'password': os.getenv('postgres_password'), # add password
}

#define the event name
EVENT_NAME = "Promotions"
#define start date_time-Control
START_DT_C = "2023-08-18"
#defined end date_time-Control
END_DT_C = "2023-08-24"
#define start date_time-Treatment
#First date of treatement is assumed to the start of the event
START_DT_T = "2023-08-25"
#defined end date_time-Treatment
END_DT_T = "2023-08-31" 
start_hour, end_hour = '06:00:00', '06:00:00' 
# event date should be a valid date between start and end date
# It is recommended that the control and treatment period is equal
use_weekly_range = True

incl_corrugate = (False,1.3)

#should the analysis control for changes in inventory position
include_inv_features = True
#should the analysis control for changes in routes
include_routes_features = True

drop_columns = []

#for normal items, currently the analysis is only for normal items
location_type = 'Chewy'    
itemtype = 'NORMAL' 

#use predicted cost outcomes
use_pred = False
use_shipping_inspector = False

#Causal Model Parameters:
# Should only the causal analysis be carried out or both data querying and causal analysis?
# if the data is already queried, skip directly to causal analysis by
steps_options = {"data_query+causal_model":"all",
                 "causal_model_only": "causal_model_only",
                 "data_query_only": "data_only"}
steps = steps_options.get("data_query+causal_model","all")
#steps = steps_options.get("causal_model_only","all")
#steps = steps_options.get("data_query_only","all")

#filter data-- IAC is applicable only for orders with less than 67lbs weight
#write a query to filter data, follow the syntax of pd.query method
data_filter = None
#data_filter = 'batch_hour<6'

#Propensity score model
#one of two models: Logistic_Regression or XgBoost
propensity_score_model_options = {"Logistic_Regression":"LReg",
                                  "XgBoost": "xgb"}
propensity_score_model = propensity_score_model_options.get("Logistic_Regression","LReg")
#propensity_score_model = propensity_score_model_options.get("XgBoost","LReg")

#number of XgBoost estiomators for Regression Model
n_estimator_reg = 200 #100, 200, 500 are all different options
max_depth_Xgb_reg = 4 #1,2,3,4,5,6,... are some alternative options
n_estimator_cl = 1
estimator_DML_dict = {'ATTE':'ATTE','ATE':'ATE'}
estimator_DML = estimator_DML_dict.get('ATE','ATE')


