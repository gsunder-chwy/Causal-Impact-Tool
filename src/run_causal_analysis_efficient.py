from configs import DATA_PATH, SQL_PATH, VERTICA_PROD_INFO, OUTPUT_PATH, POSTGRESS_PROD_INFO
from configs import (EVENT_NAME, START_DT_C, END_DT_C, START_DT_T, END_DT_T, use_weekly_range, start_hour, end_hour,
                     location_type, itemtype, data_filter, include_inv_features, include_routes_features, drop_columns)
from configs import steps, propensity_score_model, n_estimator_reg, max_depth_Xgb_reg, n_estimator_cl, estimator_DML, \
    incl_corrugate
from configs import use_pred, use_shipping_inspector
from database import Database

import pandas as pd
from utils import IO, Utils, Feature, log
import time
import numpy as np
from datetime import datetime, timedelta
import shutil

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore", message='divide by zero encountered in double_scalars')
warnings.filterwarnings('ignore',
                        message="The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.")

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

import os
import multiprocessing

from doubleml import DoubleMLData
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, Lasso, LogisticRegression
from doubleml import DoubleMLPLR, DoubleMLIRM
from lightgbm import LGBMRegressor
import glob


def causal_model(RESULT_PATH, EVENT_NAME, propensity_score_model, n_estimator_reg, max_depth_Xgb_reg):
    # df = pd.read_parquet(f'{RESULT_PATH}/')
    # df.to_parquet(f'{RESULT_PATH}/{Y}_final.parquet')

    df = []
    file_list = glob.glob(f'{RESULT_PATH}/final_data_*.parquet')
    for f in file_list:
        if isinstance(df, list):
            df = pd.read_parquet(f)
        else:
            df = pd.concat([df, pd.read_parquet(f)])

    if use_shipping_inspector and use_pred:
        log.info(f"Using Predictions for shipping inspector")
        df = df.rename(columns={"cpo": "cpo_invoice", "cpo_est_shipment_inspector": "cpo"})
        log.info(f"New Column names: {df.columns}")
    elif use_shipping_inspector:
        log.info(f"Using shipping inspector")
        df = df.rename(columns={"cpo": "cpo_invoice", "cpo_shipment_inspector": "cpo",
                                "ship_weight": "ship_weight_invoice", "ship_weight_si": "ship_weight",
                                "UPO": "UPO_invoice", "UPO_si": "UPO"})
        log.info(f"New Column names: {df.columns}")
    elif use_pred:
        log.info(f"Using Predictions")
        df = df.rename(columns={"cpo": "cpo_invoice", "cpo_predicted": "cpo",
                                "num_shipments_order": "num_shipments_order_invoice",
                                "num_shipments_order_pred": "num_shipments_order",
                                "UPO": "UPO_invoice", "UPO_pred": "UPO",
                                "cpp": "cpp_invoice", "cpp_predicted": "cpp",
                                "num_shipments_order_fedex": "num_shipments_order_fedex_invoice",
                                "num_shipments_order_ontrac": "num_shipments_order_ontrac_invoice",
                                "num_shipments_order_pred_fedex": "num_shipments_order_fedex",
                                "num_shipments_order_pred_ontrac": "num_shipments_order_ontrac",
                                "max_ship_zone": "max_ship_zone_invoice",
                                "max_ship_zone_exp": "max_ship_zone",
                                "avg_ship_zone": "avg_ship_zone_invoice",
                                "avg_ship_zone_exp": "avg_ship_zone",
                                "ship_weight": "ship_weight_invoice", "ship_weight_predicted": "ship_weight",
                                "dim_weight": "dim_weight_invoice", "dim_weight_pred": "dim_weight",
                                "actual_weight": "actual_weight_invoice", "actual_weight_pred": "actual_weight"})
        log.info(f"New Column names: {df.columns}")

    if data_filter:
        log.info(f"Filtering data based on criteria provided {data_filter}")
        log.info(f"Size of data before filter {df.shape[0]}")
        df = df.query(f'{data_filter}')
        log.info(f"Size of data after filter {df.shape[0]}")

    end_hour_int = int(datetime.strptime(end_hour, '%H:%M:%S').strftime("%H"))
    df[f'{EVENT_NAME}'] = np.where((df.date > EVENT_DATE) | ((df.date == EVENT_DATE) & (df.batch_hour >= end_hour_int)),
                                   1, 0)

    log.info(
        f'Control Period: Number of with no invoice information {df[(df.num_null_invoice == 1) & (df[f"{EVENT_NAME}"] == 0)].shape[0]}')
    log.info(
        f'Treatment Period: Number of with no invoice information {df[(df.num_null_invoice == 1) & (df[f"{EVENT_NAME}"] == 1)].shape[0]}')

    log.info(f"dropping {df.shape[0] - df.dropna().shape[0]} rows due to NA values")

    df["handling_surch_ind"] = np.where(df.handling_surch > 0, 1, 0)
    df["residential_surch_ind"] = np.where(df.residential_surch > 0, 1, 0)
    df["min_zone_has_inv"] = (df.min_zone == df.min_zone_inv).astype(int)
    log.info(df.percent_inv_lzfc.describe())
    df, _ = Feature.get_dow_variables(df)
    df["dow_weekend"] = np.where((df.dow == "Friday") | (df.dow == "Saturday") | (df.dow == "Sunday"), 1, 0)
    order_count_days = df.groupby("date", as_index=False).order_id.agg({"num_orders_day": "count"})
    df = df.merge(order_count_days, on="date")
    columns = ["date", "batch_id", "order_id", "actual_weight",
               "ship_weight", "dim_weight",
               "oversize_flag", "handling_flag",
               "dow_weekend", "num_shipments_order", "num_orders_day", "UPO", "singles",
               "num_shipments_order_ontrac", "num_shipments_order_fedex",
               # "regions","states",
               "batch_hour",
               "min_zone", "num_lz_fc",
               "ship_complete_assort_lzfc", "num_lz_fc_assort_shipcomplete", "lz_fc_assort_all_fc",
               "min_zone_inv", "ship_complete_inv_lzfc", "num_lz_fc_inv_shipcomplete", "lz_fc_inv_all_fc",
               "avg_ship_zone", "max_ship_zone",
               "num_null_invoice", "handling_surch_ind", "residential_surch_ind",
               "cpo", "cpp", "base_cost", "fuel_surch", "handling_surch", "residential_surch", "percent_inv_lzfc",
               "min_zone_has_inv"]
    columns.append(EVENT_NAME)
    df = df[columns]

    log.info(f"Details of NA {df.isna().sum()}")
    df = df.dropna()

    df["order_id"] = df.order_id.astype(int)

    if incl_corrugate[0]:
        df["cpo"] = df["cpo"] + incl_corrugate[1] * df.num_shipments_order
        df["cpp"] = df.cpp + incl_corrugate[1]

    log.info(f"{'#' * 50}")
    log.info("Data Quality Checks")
    log.info(f"Zero Actual Weight Orders {(df.actual_weight == 0).sum()}")
    log.info(f"Zero Ship Weight Orders {(df.ship_weight == 0).sum()}")
    log.info(f"Filtering Zero weight orders")
    df = df.query("(ship_weight>0) & (actual_weight>0)")
    log.info(f"Zero UPO Orders {(df.UPO == 0).sum()}")
    log.info(f"Ship Wight Smaller than order weight {((df.ship_weight - df.actual_weight) <= -1).sum()}")
    log.info(f"Filtering orders where ship weight is less than actual weight")
    old_shape = df.shape[0]
    df = df[~((df.ship_weight - df.actual_weight) <= -1)]
    # log.info(f"Dropped {old_shape-df.shape[0]} rows")
    log.info(f"{'#' * 50}")

    log.info(
        f'Control Period: Removing orders with no cost, potentially cancelled orders {df[(df.cpo == 0) & (df[f"{EVENT_NAME}"] == 0)].shape[0]}')
    log.info(
        f'Treatment Period: Removing orders with no cost, potentially cancelled orders {df[(df.cpo == 0) & (df[f"{EVENT_NAME}"] == 1)].shape[0]}')

    df = df.query("cpo>0")

    log.info(
        f'Difference in UPO between treatment and control periods {df.query(f"{EVENT_NAME}==1")["UPO"].mean() - df.query(f"{EVENT_NAME}==0")["UPO"].mean()}')

    # log.info(
    #    f'Difference in Avg ShipZone between treatment and control periods {df.query(f"{EVENT_NAME}==1")["avg_ship_zone"].mean() - df.query(f"{EVENT_NAME}==0")["avg_ship_zone"].mean()}')

    METRICS = ["cpo", "cpp"]

    for m in METRICS:
        log.info(
            f'{m}: Simple differece in mean between treatement- and control-periods {df.query(f"{EVENT_NAME}==1")[m].mean() - df.query(f"{EVENT_NAME}==0")[m].mean()}')
        if m == 'cpp':
            log.info(
                f'{m}: Per day savings between between treatement- and control-periods {(df.query(f"{EVENT_NAME}==1")[m].mean() - df.query(f"{EVENT_NAME}==0")[m].mean()) * (df.num_shipments_order.mean() * df.order_id.nunique() / df.date.nunique())}')
            log.info(
                f'{m}: Annulaized savings between between treatement- and control-periods {(df.query(f"{EVENT_NAME}==1")[m].mean() - df.query(f"{EVENT_NAME}==0")[m].mean()) * (365 * df.num_shipments_order.mean() * df.order_id.nunique() / df.date.nunique())}')
        else:
            log.info(
                f'{m}: Per day savings between between treatement- and control-periods {(df.query(f"{EVENT_NAME}==1")[m].mean() - df.query(f"{EVENT_NAME}==0")[m].mean()) * (df.order_id.nunique() / df.date.nunique())}')
            log.info(
                f'{m}: Annulaized savings between between treatement- and control-periods {(df.query(f"{EVENT_NAME}==1")[m].mean() - df.query(f"{EVENT_NAME}==0")[m].mean()) * (365 * df.order_id.nunique() / df.date.nunique())}')
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(x=df.query(f"{EVENT_NAME}==1")[m].to_numpy(),
                            y=df.query(f"{EVENT_NAME}==0")[m].to_numpy(),
                            alternative='less')
        log.info(f"MannWhitneyU test results for {m} {p}")

    log.info(
        f'Total Number of Orders: {df.shape[0]} and difference in the number of orders between the treatement- and control-periods {df.query(f"{EVENT_NAME}==1").shape[0] - df.query(f"{EVENT_NAME}==0").shape[0]}')
    '''
    _,states,regions = Feature.zipcode_to_region(df.zipcode)
    df["regions"] = regions 
    df["states"] = states
    '''

    log.info('Starting Causal Analysis')
    log.info(f'{"#" * 50}')
    log.info(
        'Time for a coffee break! Causal Analysis of CPO and CPP outcome metrics will take at least 1 hour to complete')

    log.info(df[["date", f"{EVENT_NAME}"]].drop_duplicates())

    df["ship_weight_scaled"] = (df.ship_weight - np.mean(df.ship_weight)) / np.std(df.ship_weight)
    df["UPO_scaled"] = (df.UPO - np.mean(df.UPO)) / np.std(df.UPO)
    df["batch_hour_scaled"] = (df.batch_hour - np.mean(df.batch_hour)) / np.std(df.batch_hour)
    df["actual_weight_scaled"] = (df.actual_weight - np.mean(df.actual_weight)) / np.std(df.actual_weight)
    df["dim_weight_scaled"] = (df.dim_weight - np.mean(df.dim_weight)) / np.std(df.dim_weight)
    df["carrier_fedex"] = np.where(df.num_shipments_order_fedex > 0, 1, 0)
    df["carrier_ontrac"] = np.where(df.num_shipments_order_ontrac > 0, 1, 0)

    # print(df.shape)
    # states_dummies = pd.get_dummies(df.states, prefix="state")
    # df = pd.concat([df,states_dummies],axis="columns")
    # states_cols = states_dummies.columns

    features_to_incl = ['actual_weight_scaled',
                        'dim_weight_scaled',
                        'ship_weight_scaled',
                        'dow_weekend',
                        # 'num_shipments_order_fedex','num_shipments_order_ontrac',
                        'carrier_fedex', 'carrier_ontrac',
                        'UPO_scaled',
                        'singles', 'batch_hour_scaled']
                        # ,'handling_surch_ind','residential_surch_ind']
    # features_to_incl.extend(states_cols)
    if include_routes_features:
        features_to_incl.extend(['min_zone', 'num_lz_fc'])
    if include_inv_features:
        features_to_incl.extend(['ship_complete_assort_lzfc', 'num_lz_fc_assort_shipcomplete', 'lz_fc_assort_all_fc',
                                 'min_zone_has_inv', 'min_zone_inv', 'ship_complete_inv_lzfc',
                                 'num_lz_fc_inv_shipcomplete',
                                 'lz_fc_inv_all_fc', 'percent_inv_lzfc'])
    # if not is_ORS_feature:
    #    features_to_incl.extend(['avg_ship_zone','max_ship_zone'])

    if len(drop_columns) > 0:
        features_to_incl = [f for f in features_to_incl if f not in drop_columns]

    dml_data_cpp = DoubleMLData(df, y_col='cpp', d_cols=f'{EVENT_NAME}',
                                x_cols=features_to_incl)

    # ml_l_xgb = XGBRegressor(objective = "reg:squarederror", #"reg:absoluteerror"
    #                       n_estimators = n_estimator_reg, max_depth=max_depth_Xgb_reg)

    ml_l_xgb = LGBMRegressor(objective="regression",  # "reg:absoluteerror"
                             n_estimators=n_estimator_reg, max_depth=max_depth_Xgb_reg)

    if propensity_score_model == 'xgb':
        log.info("Using Xgb Classifier as the Propensity Score Model")
        ml_m_xgb = XGBClassifier(objective="binary:logistic",
                                 eval_metric="logloss",
                                 n_estimators=n_estimator_cl)
    else:
        log.info("Using Logistic Regression as the Propensity Score Model")
        ml_m_xgb = LogisticRegression(penalty="l1", solver="saga", max_iter=5000)

    df.to_csv(f"{RESULT_PATH}/model_data.csv")

    dml_plr_tree = DoubleMLIRM(dml_data_cpp,
                               ml_g=ml_l_xgb,
                               ml_m=ml_m_xgb,
                               n_folds=5,
                               n_rep=1,
                               score=estimator_DML,
                               dml_procedure='dml2',
                               trimming_threshold=0.01)
    dml_plr_tree.fit()

    log.info(dml_plr_tree)

    '''
    est = SparseLinearDRLearner()
    est.fit(df.cpp, df[f'{EVENT_NAME}'], X=df[features_to_incl], W=None)
    log.info(est.ate(X=df[features_to_incl],T0=np.where(df[f'{EVENT_NAME}']==0,1,0),
                     T1=np.where(df[f'{EVENT_NAME}']==1,1,0)))
    '''
    log.info('Done with CPP analysis')

    log.info('Starting with CPO analysis')

    features_to_incl.extend(['num_shipments_order', 'num_shipments_order_fedex', 'num_shipments_order_ontrac'])
    dml_data_cpo = DoubleMLData(df, y_col='cpo', d_cols=f'{EVENT_NAME}',
                                x_cols=features_to_incl)

    dml_plr_tree = DoubleMLIRM(dml_data_cpo,
                               ml_g=ml_l_xgb,
                               ml_m=ml_m_xgb,
                               n_folds=5,
                               n_rep=1,
                               score=estimator_DML,
                               dml_procedure='dml2',
                               trimming_threshold=0.01)
    dml_plr_tree.fit()

    log.info(dml_plr_tree)


def data_pipeline(dates):
    if not isinstance(dates[0], list):
        dates = [dates]
    log.info("Reading Order Information")
    log.info(f'{"#" * 50}')
    start_time = time.time()

    sql = ['; '.join([IO.read_txt(f'{SQL_PATH}/order_info.sql', line_sep='\n')]) for _ in range(len(dates))]

    parameters = [{'start_date_dttm': start_date + ' ' + start_hour, 'end_date_dttm': end_date + ' ' + end_hour,
                   'start_date': start_date, 'end_date': end_date} for start_date, end_date in dates]

    with multiprocessing.Pool(processes=min(2, (multiprocessing.cpu_count() - 1))) as pool:
        result = pool.starmap_async(Database.vertica_sql_query,
                                    zip(dates, sql, parameters, [VERTICA_PROD_INFO] * len(dates)))
        pool.close()
        pool.join()

    results = result.get()
    df = pd.concat(results)
    for i, d, df_ in zip(range(len(dates)), dates, results):
        start_date, end_date = d
        file = f'{OUTPUT_PATH}/data_{start_date}_{end_date}.parquet'
        if len(df_) > 0:
            df_['date'] = pd.to_datetime(df_['date'])
            # int96 to ensure that Athena can understand the timestamp column
            df_.to_parquet(file, engine='fastparquet', times='int96')
        else:
            log.info(f'no data for time period [{start_date}, {end_date}]')
        log.info(
            f'Finished chunk {i + 1}: [{start_date}, {end_date}]: contains {df_.shape[0]} rows and took {int(time.time() - start_time)}s')

    log.info(f'{"#" * 50}')

    # key inputs
    # 1) lz_FC
    # 2) # of lz_FC that can ship complete order  (Does any of the lz FC have the assotment to ship complete)
    # 3) can all parts potentially be shipped from lz_FC? does all lz_FC put together have the assortment to fulfill the order
    # 4) Singles? Singles may not be fulfilled from lz_FC
    # 4) lz_FC with Inventory
    # 5) lz_inv ship complete

    # load queried data
    # df = pd.read_parquet(os.path.abspath(f'{OUTPUT_PATH}'))
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")
    """
    #since the query is broken down over multiple weeks, it is possible to have duplicates if an order is processed over multiple weeks
    # annectdotal example: order_id 1103306175 was in the order_items_to_routes table twice: once on the 19th and again on the 22nd of February 
    # find duplicates
    duplicated_mask = df.duplicated('order_id',keep=False)
    df_non_duplicated = df[~duplicated_mask]
    #check if all the duplicates have the cpo values, if yes, dropping makes sense. It is possible that the edge cases at the border of the week splits might cause problems
    df_duplicated = df[duplicated_mask]
    duplicated_cpo = df_duplicated.groupby("order_id",as_index=False).apply(lambda z: pd.Series({'cpo_non_identical':z.cpo.nunique()!=1}))
    duplicated_cpo = pd.merge(df_duplicated,duplicated_cpo,on="order_id")
    duplicated_cpo = pd.concat([duplicated_cpo[duplicated_cpo.cpo_non_identical],duplicated_cpo[~duplicated_cpo.cpo_non_identical].drop_duplicates("order_id",keep="last")])
    df = pd.concat([df_non_duplicated,duplicated_cpo.drop("cpo_non_identical",axis="columns")])

    df = df.sort_values("date")
    """
    # simple fix any duplicates because the batch was processed over multiple batches use only the last batch
    df = df.drop_duplicates("order_id", keep="last")

    # get FC information
    # sql = IO.read_txt(f'{SQL_PATH}/locations.sql', '')
    sql = f'''
            SELECT location_key,
                   location_code -- fulfillment center's name
            FROM chewybi.locations
            WHERE
                (
                 fulfillment_active = 'true'
                 --OR fulfillment_active IS NULL
                )
            AND (
            location_warehouse_type = 0
            OR location_warehouse_type IS NULL
            )
            AND location_active_warehouse = 1
            -- 'Chewy' for core network
            -- 'Chewy Pharmacy' for pharmacy
            -- 'Chewy Healthcare Services' for healthcare 
            AND product_company_description = '{location_type}'
            AND location_code NOT IN ('ITM0') 
            UNION
            SELECT location_key,
            location_code -- fulfillment center's name
            FROM
            chewybi.locations
            WHERE location_code IN ('WFC2') -- hardcoded for including WFC2 in the early part of the year
            ;
           '''

    assert location_type in ['Chewy', 'Chewy Pharmacy',
                             'Chewy Healthcare Services'], f"Location type should be either Chewy, or Chewy Pharmacy, or Chewy Healthcare Services"

    locations = Database.vertica_get_df(VERTICA_PROD_INFO, sql, {'location_type': location_type})
    fc_list = locations["location_code"].to_list()
    # hard coding and adding WFC2 which was active in early part of 2023
    # fc_list.append('WFC2')
    fc_list = tuple(fc_list)

    log.info(f"The FCs considered in this analysis are {fc_list}")

    if itemtype == 'NORMAL':
        itemtypelist = ('N')
    else:
        raise ValueError("Cuurently tool supports only normal items")

    log.info("Querying routes information")
    log.info(f'{"#" * 50}')
    start_time = time.time()
    sql = [f"""select *,
                      min(zone) over (partition by date, zipcode) as min_zone
                  from (select distinct   date,
                                          zip5 as zipcode,
                                          fcname,
                                          --(CASE WHEN zone=1 THEN 2 ELSE zone END) as zone
                                          zone
                        from ors_simulations.ors2_routes
                        where date between date('{d[0]}') and date('{d[1]}') and fcname in {fc_list} and orsitemtype = '{itemtypelist}'
                        order by zipcode, fcname, zone) s""" for d, fc_list, itemtypelist in
           zip(dates, [fc_list] * len(dates), [itemtypelist] * len(dates))]

    with multiprocessing.Pool(processes=min(len(dates), (multiprocessing.cpu_count() - 1))) as pool:
        result = pool.starmap_async(Database.postgres_sql_query, zip(dates, sql, [POSTGRESS_PROD_INFO] * len(
            dates)))  # ,callback=lambda res: print(res,flush=True))
        pool.close()
        pool.join()

    fc_zone_details = result.get()
    fc_zone_details = pd.concat(fc_zone_details)
    fc_zone_details = fc_zone_details.drop_duplicates()

    log.info(f"Done reading routes information. Took {time.time() - start_time}s")
    log.info(f'{"#" * 50}')

    lz_fc_zip = fc_zone_details.query("zone==min_zone")[["date", "zipcode", "zone", "min_zone"]].groupby(
        ["date", "zipcode"], as_index=False).agg(min_zone=('min_zone', 'first'), num_lz_fc=('zone', 'count'))

    # assortment
    log.info(f"Querying assortment information.")
    log.info(f'{"#" * 50}')
    start_time = time.time()
    sql = [f"""select inventory_snapshot_snapshot_dt as inv_date,
                     location_code,
                     product_part_number,
                     inventory_snapshot_quantity,
                     inventory_snapshot_managed_flag,
                     inventory_snapshot_out_of_stock_eligible_flag,
                     inventory_snapshot_out_of_stock_flag,
                     item_location_product_discontinued_flag
              from chewybi.inventory_snapshot_pharmacy
              where inventory_snapshot_snapshot_dt between date('{d[0]}') and date('{d[1]}') and location_code in {fc_list}
              and item_location_product_discontinued_flag = 'false'
              """ for d, fc_list in zip(dates, [fc_list] * len(dates))]
    parameters = [{} for _ in dates]
    # with multiprocessing.Pool(processes=min(len(dates),(multiprocessing.cpu_count()-1))) as pool:
    with multiprocessing.Pool(processes=min(2, (multiprocessing.cpu_count() - 1))) as pool:
        result = pool.starmap(Database.vertica_sql_query, zip(dates, sql, parameters, [VERTICA_PROD_INFO] * len(
            dates)))
        pool.close()
        pool.join()
    assortment_details = result

    sql = [f""" with order_det1 as (
                        select  NEW_TIME(batch_dttm,'UTC','America/New_York') as batch_dttm,
                                batch_id,
                                order_id,
                                TRIM(itemtype),
                                partnumber,
                                zipcode
                        from ors.order_items_to_route
                        where date(NEW_TIME(batch_dttm,'UTC','America/New_York')) between date('{d[0]}') and date('{d[1]}') and TRIM(itemtype) in ('{itmtype}')
                        ),
                     order_det2 as (
                        select order_id,
                               COUNT(DISTINCT partnumber) as units_in_order
                               from ors.order_items_to_route
                        where date(NEW_TIME(batch_dttm,'UTC','America/New_York')) between date('{d[0]}') and date('{d[1]}') and itemtype in ('{itmtype}') 
                        group by order_id
                        ),
                     order_det3 as (
                                select order_det1.*,
                                order_det2.units_in_order,
                                max(batch_dttm) over (partition by order_det1.order_id,order_det1.partnumber) as last_batch
                    from order_det1  
                    join order_det2
                    on order_det1.order_id = order_det2.order_id
                    )
                select distinct *,
                                date(batch_dttm) as date
                from order_det3
                where batch_dttm = last_batch """ for d, itmtype in zip(dates, [itemtype] * len(dates))]
    parameters = [{} for _ in dates]

    with multiprocessing.Pool(processes=min(2, (multiprocessing.cpu_count() - 1))) as pool:
        result = pool.starmap(Database.vertica_sql_query, zip(dates, sql, parameters, [VERTICA_PROD_INFO] * len(dates)))
        pool.close()
        pool.join()
    order_details = result

    order_details_zip = [pd.merge(ord_det, fc_zone_details.query("zone==min_zone"), on=["date", "zipcode"]) for ord_det
                         in order_details]

    with multiprocessing.Pool(processes=min(len(order_details_zip), (multiprocessing.cpu_count() - 1))) as pool:
        result = pool.starmap_async(Feature.assortment_info,
                                    zip(range(len(order_details_zip)), order_details_zip, assortment_details))
        pool.close()
        pool.join()
    order_assortment = result.get()
    order_assortment = pd.concat(order_assortment)
    order_assortment = order_assortment.reset_index()
    order_assortment = order_assortment.drop_duplicates("order_id", keep="last")

    lz_fc_zip["date"] = pd.to_datetime(lz_fc_zip.date)
    df = pd.merge(df, lz_fc_zip, on=["date", "zipcode"], how="left")

    df = pd.merge(df, order_assortment, on="order_id", how="left")
    # log.info(f"Number of mismatches in order_ids with no relevant assortment information available {df.shape[0],df_wo_assort_shape}")
    order_details = pd.concat(order_details)

    log.info(f"Assortment details shape: {order_assortment.shape}")
    log.info(f"Done reading assortment information. Took {time.time() - start_time}s")
    log.info(f'{"#" * 50}')

    log.info(f"Querying inventory information.")
    log.info(f'{"#" * 50}')
    start_time = time.time()
    date_list = list(order_details.batch_dttm.dt.date.unique())
    date_list.sort()
    batch_list = []
    order_details_list = []
    for dt in date_list:
        order_details_list.append(order_details[order_details.batch_dttm.dt.date == dt])
        batch_list.append(tuple(order_details_list[-1]["batch_id"].unique()))
        # check logic with time, UTC vs EST time difference is messing things up
    sql = [f"""select *
                    from ors_simulations.ors2_athena_inventory_snapshot 
                    where batch_id in {batches} and 
                    fc_name in {fc_list}
           """ for batches, fc_list in zip(batch_list, [fc_list] * len(batch_list))]

    num_cores = (multiprocessing.cpu_count()) // 2
    with multiprocessing.Pool(processes=num_cores) as pool:
        result = pool.starmap(Database.postgres_sql_query,
                              zip(date_list, sql, [POSTGRESS_PROD_INFO] * (len(date_list))))
        pool.close()
        pool.join()
    batch_inv_details = result

    inv_details_list = []
    for dt, batch_inv_det, order_det in zip(date_list, batch_inv_details, order_details_list):
        log.info(f"Processing Date {dt}")
        order_det = order_det.astype({"partnumber": int})
        # batch_inv_det = batch_inv_det.astype({"partnumber":str})
        batch_inv_det["fc_name"] = batch_inv_det["fc_name"].apply(lambda z: z.strip())
        batch_inv_det = pd.merge(order_det, batch_inv_det[["batch_id", "fc_name", "partnumber", "quantity_available"]],
                                 on=["batch_id", "partnumber"], how="left")
        batch_inv_det = batch_inv_det.query("quantity_available>0")
        batch_inv_det = pd.merge(batch_inv_det, fc_zone_details, left_on=["date", "zipcode", "fc_name"],
                                 right_on=["date", "zipcode", "fcname"], how="left")
        batch_inv_det = batch_inv_det.astype({"fc_name": "category"})
        batch_inv_det["fc_name_code"] = batch_inv_det["fc_name"].cat.codes
        inv_details_list.append(
            batch_inv_det[["order_id", "fc_name_code", "zone", "units_in_order", "partnumber", "quantity_available"]])

    # issue of running out of memory in this step: break it into multiple smaller list and close the multiprocess pool
    #inventory_details = None
    num_cores = (multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_cores) as pool:
        result = pool.map_async(Feature.inv_details, inv_details_list, chunksize=None)
        pool.close()
        pool.join()
    inventory_details_ = result.get()
    inventory_details = pd.concat(inventory_details_)

    #log.info(f"Inventory details shape: {inventory_details.shape}")
    #log.info(
    #    f"Discrepancy of {order_assortment.shape[0] - inventory_details.shape[0]} in the unique orders between assortment and inventory is because the were not fulfilled from the chosen FCs")
    log.info(f"Done reading inventory information. Took {time.time() - start_time}s")
    log.info(f'{"#" * 50}')

    inventory_details = inventory_details.drop_duplicates("order_id", keep="last")
    df = pd.merge(df, inventory_details, on="order_id", how="left")
    df["singles"] = np.where(df.units_in_order == 1, 1, 0)

    df['date'] = pd.to_datetime(df['date'])
    # int96 to ensure that Athena can understand the timestamp column
    start_date = dates[0][0]
    end_date = dates[-1][-1]
    df.to_parquet(f'{RESULT_PATH}/final_data_{start_date}_{end_date}.parquet', engine='fastparquet', times='int96')


if __name__ == '__main__':
    START_DATE_T, END_DATE_T = pd.to_datetime(START_DT_T).strftime("%Y-%m-%d"), pd.to_datetime(END_DT_T).strftime(
        "%Y-%m-%d")
    START_DATE_C, END_DATE_C = pd.to_datetime(START_DT_C).strftime("%Y-%m-%d"), pd.to_datetime(END_DT_C).strftime(
        "%Y-%m-%d")
    EVENT_DATE = START_DATE_T
    # basic checks
    assert START_DATE_T < END_DATE_T, f"Input Error: Start Date of treatment group {START_DATE_T} is on or after End Date {END_DATE_T}"
    assert START_DATE_C < END_DATE_C, f"Input Error: Start Date of control group {START_DATE_C} is on or after End Date {END_DATE_C}"
    assert START_DATE_T >= END_DATE_C, f"Input Error: Overlap in treatment and control date ranges"

    Y = 'cpo'
    OUTPUT_PATH = f'{DATA_PATH}/{Y}_{EVENT_NAME}_{START_DT_C}_{END_DT_T}'
    RESULT_PATH = f'{OUTPUT_PATH}/{Y}'

    if steps == "causal_model_only":
        causal_model(RESULT_PATH, EVENT_NAME, propensity_score_model, n_estimator_reg, max_depth_Xgb_reg)
        exit()

    # clean up the data directory
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
    os.makedirs(OUTPUT_PATH)

    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH, ignore_errors=True)
    os.makedirs(RESULT_PATH)

    if use_weekly_range:
        dates = Utils.get_weekly_date_ranges(START_DATE_C, END_DATE_C, '%Y-%m-%d')
        dates.extend(Utils.get_weekly_date_ranges(START_DATE_T, END_DATE_T, '%Y-%m-%d'))
    else:
        dates = [datetime.strptime(START_DATE_C, '%Y-%m-%d').strftime('%Y-%m-%d'),
                 datetime.strptime(END_DATE_C, '%Y-%m-%d').strftime('%Y-%m-%d'),
                 datetime.strptime(START_DATE_T, '%Y-%m-%d').strftime('%Y-%m-%d'),
                 datetime.strptime(END_DATE_T, '%Y-%m-%d').strftime('%Y-%m-%d')]

    dates_list = [dates[idx:(idx + 2)] for idx in range(0, len(dates), 2)]

    log.info(f"Evaluating for dates {dates_list}")

    for d in dates_list:
        data_pipeline(d)

    if steps == "data_only":
        exit()

    causal_model(RESULT_PATH, EVENT_NAME, propensity_score_model, n_estimator_reg, max_depth_Xgb_reg)
