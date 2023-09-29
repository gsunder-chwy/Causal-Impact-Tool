from configs import VERTICA_PROD_INFO
import vertica_python
import pandas as pd
from utils import log
import psycopg2

class Database:
    
    @staticmethod
    def _db_connect(database, info):
        def core(func):
            def with_connection_(*args, **kwargs):
                if database == 'vertica':
                    f = vertica_python
                    log.info('database connection to vertica successful')
                elif database == 'postgres':
                    f = psycopg2
                    log.info('database connection to postgres successful')
                cn = f.connect(**info)
                try:
                    rv = func(cn, *args,**kwargs)
                except Exception:
                    if database != 'athena':
                        cn.rollback()
                    log.error('database connection error')
                    raise
                else:
                    if database != 'athena':
                        cn.commit()
                finally:
                    cn.close()
                    log.info('database closed')
                return rv

            return with_connection_
        return core

    @staticmethod
    def vertica(info):
        return Database._db_connect('vertica', info)
    
    @staticmethod
    def postgres(info):
        return Database._db_connect('postgres', info)
    
    @staticmethod
    def vertica_get_df(cn_info, sql, parameters={}, callback=None):

        @Database.vertica(info=cn_info)
        def _get_df(cn, sql, parameters={}, callback=None):
            data = []
            with cn.cursor('dict') as cur:
                cur.execute(sql, parameters)
                # loop thru each query and collect results
                while True:
                    cache = [ callback(row) if callback else row for row in cur.iterate() ]
                    if len(cache) > 0:
                        data.append(pd.DataFrame(cache))
                    if not cur.nextset():
                        break
            # if there is only one resultset, then return the pandas
            # else return a list of pandas
            return data[0] if len(data) == 1 else data

        return _get_df(sql, parameters, callback)

    @staticmethod
    def postgres_get_df(cn_info, sql, parameters={}, callback=None):

        @Database.postgres(info=cn_info)
        def _get_df(cn, sql, parameters={}, callback=None):
            data = []
            with cn.cursor('dict') as cur:
                cur.execute(sql)
                data = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]
                data = pd.DataFrame(data, columns=colnames)
            return data
        
        return _get_df(sql, parameters, callback)
    
    def vertica_sql_query(d,sql,parameters,VERTICA_PROD_INFO):
        log.info(f'Starting the SQL query for dates {d}')
        df = Database.vertica_get_df(
                    VERTICA_PROD_INFO, 
                    sql, 
                    parameters=parameters
            )
        return df
    
    def postgres_sql_query(d,sql,POSTGRESS_PROD_INFO):
        log.info(f'Starting the SQL query for dates {d}')
        df = Database.postgres_get_df(POSTGRESS_PROD_INFO, sql)
        return df


if __name__ == '__main__':

    # dummy parameters
    START_DATE = '2022-05-01'
    END_DATE = '2022-09-30'

    # vertica
    df = Database.vertica_get_df(
        VERTICA_PROD_INFO, 
        'SELECT * FROM sandbox_sc_network.ors_events ORDER BY valid_from, valid_to', 
        parameters={ 
            'start_date': START_DATE, 
            'end_date': END_DATE 
        }
    )
    log.info(df)