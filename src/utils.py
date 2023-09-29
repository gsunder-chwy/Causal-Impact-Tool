import logging
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import itertools
import hashlib
import uszipcode

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from configs import EVENT_NAME,START_DT_C,END_DT_T

import multiprocessing as mp

import matplotlib
matplotlib.interactive(False)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
# %matplotlib inline  

import warnings
warnings.filterwarnings("ignore", message='invalid value encountered in double_scalars')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logformat = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"                 
formatter = logging.Formatter(fmt=logformat)               
file_handler = logging.FileHandler(filename=f'{EVENT_NAME}_{START_DT_C}_{END_DT_T}.log')          
file_handler.setFormatter(formatter)                                            
log.addHandler(file_handler)                                                 
console = logging.StreamHandler()                                               
console.setFormatter(formatter)                                                 
log.addHandler(console)

class IO:
    
    @staticmethod
    def read_txt(path, line_sep=None):
        lines = []
        with open(path, 'r') as f:
            lines = f.readlines()
        return line_sep.join(lines) if line_sep != None else lines
            
class Feature:
    
    @staticmethod
    def get_dow_variables(df):
        # add dow
        dow_vars = []
        dow_cols = {}
        df['dow'] = df['date'].dt.day_name()
        for dow in ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            name = f'{dow.lower()}_ind'
            dow_cols[name] = (df['dow'] == dow).astype(int)
            dow_vars.append(name)
        df = pd.concat([df, pd.DataFrame(dow_cols)], axis=1)
        return df, dow_vars
    
    @staticmethod
    def zipcode_to_region(zip_column):
        log.info("Mapping Zipcode information to State ")
        states = {
                    'AK': 'Alaska',
                    'AL': 'Alabama',
                    'AR': 'Arkansas',
                    'AZ': 'Arizona',
                    'CA': 'California',
                    'CO': 'Colorado',
                    'CT': 'Connecticut',
                    'DC': 'District of Columbia',
                    'DE': 'Delaware',
                    'FL': 'Florida',
                    'GA': 'Georgia',
                    'HI': 'Hawaii',
                    'IA': 'Iowa',
                    'ID': 'Idaho',
                    'IL': 'Illinois',
                    'IN': 'Indiana',
                    'KS': 'Kansas',
                    'KY': 'Kentucky',
                    'LA': 'Louisiana',
                    'MA': 'Massachusetts',
                    'MD': 'Maryland',
                    'ME': 'Maine',
                    'MI': 'Michigan',
                    'MN': 'Minnesota',
                    'MO': 'Missouri',
                    'MS': 'Mississippi',
                    'MT': 'Montana',
                    'NC': 'North Carolina',
                    'ND': 'North Dakota',
                    'NE': 'Nebraska',
                    'NH': 'New Hampshire',
                    'NJ': 'New Jersey',
                    'NM': 'New Mexico',
                    'NV': 'Nevada',
                    'NY': 'New York',
                    'OH': 'Ohio',
                    'OK': 'Oklahoma',
                    'OR': 'Oregon',
                    'PA': 'Pennsylvania',
                    'RI': 'Rhode Island',
                    'SC': 'South Carolina',
                    'SD': 'South Dakota',
                    'TN': 'Tennessee',
                    'TX': 'Texas',
                    'UT': 'Utah',
                    'VA': 'Virginia',
                    'VT': 'Vermont',
                    'WA': 'Washington',
                    'WI': 'Wisconsin',
                    'WV': 'West Virginia',
                    'WY': 'Wyoming'
                }
        states_to_regions = {
                            'Washington': 'West', 'Oregon': 'West', 'California': 'West', 'Nevada': 'West',
                            'Idaho': 'West', 'Montana': 'West', 'Wyoming': 'West', 'Utah': 'West',
                            'Colorado': 'West', 'Alaska': 'West', 'Hawaii': 'West', 'Maine': 'Northeast',
                            'Vermont': 'Northeast', 'New York': 'Northeast', 'New Hampshire': 'Northeast',
                            'Massachusetts': 'Northeast', 'Rhode Island': 'Northeast', 'Connecticut': 'Northeast',
                            'New Jersey': 'Northeast', 'Pennsylvania': 'Northeast', 'North Dakota': 'Midwest',
                            'South Dakota': 'Midwest', 'Nebraska': 'Midwest', 'Kansas': 'Midwest',
                            'Minnesota': 'Midwest', 'Iowa': 'Midwest', 'Missouri': 'Midwest', 'Wisconsin': 'Midwest',
                            'Illinois': 'Midwest', 'Michigan': 'Midwest', 'Indiana': 'Midwest', 'Ohio': 'Midwest',
                            'West Virginia': 'South', 'District of Columbia': 'South', 'Maryland': 'South',
                            'Virginia': 'South', 'Kentucky': 'South', 'Tennessee': 'South', 'North Carolina': 'South',
                            'Mississippi': 'South', 'Arkansas': 'South', 'Louisiana': 'South', 'Alabama': 'South',
                            'Georgia': 'South', 'South Carolina': 'South', 'Florida': 'South', 'Delaware': 'South',
                            'Arizona': 'Southwest', 'New Mexico': 'Southwest', 'Oklahoma': 'Southwest',
                            'Texas': 'Southwest'
                            }
        search = uszipcode.search.SearchEngine()
        state_code = []
        for idx,z in enumerate(zip_column):
            if idx%1000000==0:
                log.info(f"Completed {idx}/{zip_column.shape[0]} rows")
            state_code.append(search.by_zipcode(z))
        states_code_ = [states[s.state] for s in state_code]
        regions = [states_to_regions[s] for s in states_code_]
        return states, states_code_, regions
    
    def assortment_info(idx,ordr_details_zip,assort_details):
        print(f"Getting Assortment information for week {idx}")
        ordr_details_zip["batch_date"] = ordr_details_zip.batch_dttm.dt.date 
        order_assortment_details = pd.merge(ordr_details_zip,assort_details,left_on=["batch_date","partnumber","fcname"],
                                        right_on=["inv_date","product_part_number","location_code"],
                                        how="left") 
        order_assortment_details = order_assortment_details.astype({"partnumber":int,"fcname":"category","product_part_number":float})
        order_assortment_details["fcname_code"] = order_assortment_details.fcname.cat.codes
        group_agg = order_assortment_details[["order_id","product_part_number","fcname_code","units_in_order"]].groupby(["order_id","fcname_code"]).agg(count_parts=("product_part_number","count"),units_in_order=("units_in_order","min"))
        group_agg["ship_complete_flag"] = group_agg.count_parts==group_agg.units_in_order
        group_agg = group_agg.groupby("order_id").agg(num_lz_fc_assort_shipcomplete=("ship_complete_flag","sum"))
        group_agg["ship_complete_assort_lzfc"] = (group_agg.num_lz_fc_assort_shipcomplete>0).astype(int)
        group_agg2 = order_assortment_details[["order_id","product_part_number","units_in_order"]].groupby("order_id").agg(count_parts=("product_part_number","nunique"),units_in_order=("units_in_order","min")).eval("lz_fc_assort_all_fc=count_parts==units_in_order").astype(int)
        group_agg = group_agg.join(
                        group_agg2["lz_fc_assort_all_fc"]
                       )
        return group_agg

    def inv_details(inv_det_df):
        group_agg_order = inv_det_df.groupby("order_id",as_index=False).agg(min_zone_inv=("zone","min"),
                                                                        count_parts=("partnumber","nunique"),
                                                                        units_in_order=("units_in_order","min"))
        #get min_zone_inv
        inv_det_df = pd.merge(inv_det_df,group_agg_order[["order_id","min_zone_inv"]],
                          on="order_id")
        #filter_by_min_zone_inv
        inv_det_df = inv_det_df.query("zone==min_zone_inv")
        #groupby fc and check for ship complete
        #group_agg = inv_det_df.groupby(["order_id","fc_name_code"]).agg(count_parts=("partnumber","count"),units_in_order=("units_in_order","min"))
        group_agg = inv_det_df.groupby(["order_id","fc_name_code"]).agg(count_parts=("partnumber","nunique"),units_in_order=("units_in_order","min"))
        group_agg["ship_complete_flag"] = group_agg.count_parts==group_agg.units_in_order
        group_agg["percent_inv_lzfc_"] = group_agg.count_parts/group_agg.units_in_order
        group_agg = group_agg.groupby("order_id").agg(num_lz_fc_inv_shipcomplete=("ship_complete_flag","sum"),
                                                      percent_inv_lzfc = ("percent_inv_lzfc_","max"))
        group_agg["ship_complete_inv_lzfc"] = (group_agg.num_lz_fc_inv_shipcomplete>0).astype(int)
        group_agg_order["lz_fc_inv_all_fc"] = group_agg_order.eval("count_parts==units_in_order").astype(int)
        group_agg = pd.merge(group_agg.reset_index(),group_agg_order,
                         on="order_id")
        return group_agg

class Utils:
    
    @staticmethod
    def get_md5(string):
        return hashlib.md5(string.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_estimation_date_ranges(event_name, events, default_start_dt=None, default_end_dt=None):
        
        if default_start_dt is not None and default_end_dt is not None:
            events = events[events['valid_from'].between(
                pd.to_datetime(default_start_dt),
                pd.to_datetime(default_end_dt)
            )].reset_index(drop=True)
            return events, default_start_dt, default_end_dt
        
        def get_dates(_events, events, n):
            tmp = events[events['event_name'].isin(_events)].reset_index(drop=True)
            start_dt = tmp['valid_from'].min()
            end_dt = tmp['valid_from'].max()
            return tmp, start_dt, end_dt
        
        INF_END_DT = pd.to_datetime('2099-12-31 00:00:00')
        _events = { event_name }
        N = 14
        while True:
            _num_events = len(_events)
            tmp = events[events['event_name'].isin(_events)].reset_index(drop=True)
            start_dt = tmp['valid_from'].min()
            end_dt = tmp['valid_from'].max()
            for i, e in events.iterrows():
                if e['event_name'] in _events:
                    continue
                if 0 <= abs(start_dt - e['valid_from']).days <= N:
                    # or 0 <= (start_dt - e['valid_to']).days <= N:
                    _events.add(e['event_name'])
                    continue
                if 0 <= abs(e['valid_from'] - end_dt).days <= N:
                #     # if e['valid_to'] == INF_END_DT \
                #     #     or 0 <= (e['valid_to'] - end_dt).days <= N:
                    _events.add(e['event_name'])
                    continue
            if _num_events == len(_events):
                break
        events = events[events['event_name'].isin(_events)].reset_index(drop=True)
        start_dt = events['valid_from'].min()
        end_dt = events['valid_to'].max()
        if end_dt == INF_END_DT:
            end_dt = events['valid_from'].max() + pd.Timedelta(days=7)
        start_dt = start_dt - pd.Timedelta(days=7)
        log.info(f'involved events: {_events}, start_dt: {start_dt}, end_dt: {end_dt}')
        if default_start_dt:
            start_dt = pd.to_datetime(default_start_dt)
        if default_end_dt:
            end_dt = pd.to_datetime(default_end_dt)
        yesterday = pd.to_datetime(datetime.now() - timedelta(days=1))
        if end_dt > yesterday:
            end_dt = yesterday
        start_dt = start_dt.strftime('%Y-%m-%d')
        end_dt = end_dt.strftime('%Y-%m-%d')
        log.info(f'start dates: {start_dt}, end date: {end_dt}')
        return events, start_dt, end_dt         
        

    @staticmethod
    def get_monthly_date_ranges(start_date, end_date, date_format):
        dates = []
        date = datetime.strptime(start_date, date_format)
        date = date.replace(day=1)
        last_date = datetime.strptime(end_date, date_format)
        while date < last_date:
            _start_date = date
            date = _start_date + relativedelta(months=1)
            _end_date = date - timedelta(days=1)
            dates.append([_start_date.strftime(date_format), _end_date.strftime(date_format)])
        # if len(dates) > 0:
        #     dates[0][0] = start_date
        #     dates[-1][1] = end_date
        return dates
    
    @staticmethod
    def get_weekly_date_ranges(start_date, end_date, date_format):
        dates = []
        date = datetime.strptime(start_date, date_format)
        #date = date.replace(day=1)
        last_date = datetime.strptime(end_date, date_format)
        while date < last_date:
            _start_date = date
            date = _start_date + relativedelta(weeks=1)
            _end_date = date #- timedelta(days=1)
            dates.append([_start_date.strftime(date_format), _end_date.strftime(date_format)])
        # if len(dates) > 0:
        #     dates[0][0] = start_date
        #     dates[-1][1] = end_date
        return dates
    
    # loop thru all parameter combinations
    def _grid_search_core(args):
        i, data, Model, params, size = args
        start_time = time.time()
        mses = []
        r2s = []
        for x_train, y_train, x_test, y_test in data:
            # this requires the model to have the same interface as sklearn
            model = Model(**params)
            model.fit(x_train, y_train)
            yhat = model.predict(x_test)
            mses.append(mean_squared_error(y_test, yhat))
            r2s.append(r2_score(y_test, yhat))
        mse, r2 = np.mean(mses), np.mean(r2s)
        log.info(', '.join(
            [ f'{int((i+1)/size * 100)}%' ] + \
            [ f'{k}: {v}' for k, v in params.items() ] + \
            [ f'mse: {round(mse, 4)}, r2: {round(r2, 4)}' ] + \
            [ f'took: {int(time.time() - start_time)}s' ]
        ))
        return mse
    
    @staticmethod
    def grid_search(X, y, Model, params, stratified=False, K=None):
        # convert { a: [1, 2], b: [3, 4] } => [ {a: 1, b: 3}, {a: 1, b: 4}, ... ]
        keys, values = zip(*params.items())
        all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # each fold has to have at least 10 observations
        N = len(X)
        if K is None:
            K = max(int(N / 10), 5)
        fold_params = {
            'n_splits': K, 
            'shuffle': True, 
            'random_state': 0
        }
        f = StratifiedKFold(**fold_params) if stratified else KFold(**fold_params)
        splits = f.split(X, y) if stratified else f.split(X)
        data = []
        for train_index, test_index in splits:
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            data.append([x_train, y_train, x_test, y_test])
        # 
        iters = [ (i, data, Model, p, len(all_params)) for i, p in enumerate(all_params) ]
        """
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            mses = list(pool.imap(Utils._grid_search_core, iters))
        """
        mses = []
        for args in iters:
            mses.append(Utils._grid_search_core(args))
        best_idx = np.argmin(mses)
        best_params = all_params[best_idx]
        log.info(f'done with {K} folds resulted the following optimal parameters:')
        log.info(f'MSE       : {round(mses[best_idx], 4)}')      
        log.info(f'Parameters: {best_params}')  
        return best_params, K
    
    @staticmethod
    def grid_search_ts(W, F, df, xvars, y, Model, params):
        splits = [ (i, i+F) for i in range(W, df.shape[0] - F + 1) ]
        K = len(splits)
        mses = []
        for i, p in enumerate(params):
            xdf = df[xvars]
            ydf = df[y]
            _mses = []
            for b, e in splits:
                try:
                    start_time = time.time()
                    model = Model(
                        endog=ydf.iloc[:b],
                        exog=xdf.iloc[:b],
                        order=(p['ar'], p['i'], p['ma']),
                        seasonal_order=(p['sar'], p['si'], p['sma'], p['s']),
                        # enforce_stationarity=False, 
                        # enforce_invertibility=False
                    ).fit()
                    yhat = model.forecast(steps=e-b, exog=xdf.iloc[b:e])
                    mse = mean_squared_error(ydf.iloc[b:e], yhat)
                    _mses.append(mse)
                    log.info(', '.join(
                        [ f'{int((i+1)/len(params) * 100)}%' ] + \
                        [ f'b: {b}, e: {e}'] + \
                        [ f'{k}: {v}' for k, v in p.items() ] + \
                        [ f'mse: {round(mse, 4)}' ] + \
                        [ f'took: {int(time.time() - start_time)}s' ]
                    ))
                except:
                    log.error('error encountered while fitting ARIMA' + \
                        f'({p["ar"]}, {p["i"]}, {p["ma"]})({p["sar"]}, {p["si"]}, {p["sma"]}, {p["s"]})')
                    _mses.append(9999999.0)
            mses.append(np.mean(_mses))
        best_idx = np.argmin(mses)
        best_params = params[best_idx]
        log.info(f'done with {K} fold resulted in the following minimal MSE and optimal parameters:')
        log.info(f'MSE       : {round(mses[best_idx], 4)}')      
        log.info(f'Parameters: {best_params}')    
        return best_params, K
        
    
            