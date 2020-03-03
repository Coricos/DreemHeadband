# Author:  Meryll Dindin
# Date:    02 March 2020
# Project: DreemEEG

from utils import *

class SqlManager:

    def __init__(self, local_path='default.db', connection_type='sqlite3', mysql_credentials=None): 

        self.fmt = connection_type

        if connection_type == 'sqlite3':
            self.dtb = local_path

        if connection_type == 'mysql':
            self.rds = mysql_credentials

    def connect(self):

        if self.fmt == 'sqlite3': 
            return sqlite3.connect(self.dtb)

        if self.fmt == 'mysql': 
            return pymysql.connect(**self.rds)

    def binary_size(self, vector):

        return len(vector.tostring())

    def populate(self, table_metas, types_metas, metas, table_times, types_times, times):

        if self.fmt == 'sqlite3':
            con = self.connect()
            metas.to_sql(table_metas, con, if_exists='append', dtype=types_metas, index=True)
            times.to_sql(table_times, con, if_exists='append', dtype=types_times, index=True)
            con.commit()
            con.close()

        if self.fmt == 'mysql':
            arg = ['user', 'password', 'host', 'port', 'database']
            con = '{}:{}@{}:{}/{}'.format(*tuple([self.rds[k] for k in arg]))
            con = create_engine('mysql+pymysql://' + con)
            metas.to_sql(table_metas, con, if_exists='append', dtype=types_metas, index=True)
            times.to_sql(table_times, con, if_exists='append', dtype=types_times, index=True)
            con.commit()
            con.close()
            
    def fetch(self, query):
        
        con = self.connect()
        cur = con.cursor()
        cur.execute(query)
        qry = cur.fetchall()
        con.close()

        key = [e[0] for e in cur.description]
        qry = [{k: v for k, v in zip(key, e)} for e in qry]

        return qry
    
    def execute(self, query):
        
        con = self.connect()
        cur = con.cursor()
        cur.execute(query)
        con.close()
    
    def ts_batch(self, table, indexes, dtype=float):
        
        idx = "(" + ','.join([str(e) for e in indexes]) + ")"
        qry = self.fetch("SELECT id, ts FROM {} WHERE id IN {}".format(table, idx))
        
        return np.asarray([np.frombuffer(e.get('ts'), dtype=np.float32) for e in qry])
    
    def ts_chunk(self, batch=cpu_count()):
        
        for i in range(0, self._ts, batch): yield list(range(i, min(i+batch, self._ts)))
    
    def ts_count(self, table):
        
        con = sql.connect()
        cur = con.cursor()
        cur.execute("SELECT COUNT(id) FROM {}".format(table))
        self._ts = cur.fetchone()[0]
        con.close()
    
    def featurize(self, table_series, table_features, frequency, batch=cpu_count()):

        self.ts_count(table_series)

        new = partial(featurize, frequency=frequency)
        con = self.connect()
        bar = tqdm.tqdm(total=(self._ts//batch)+1)

        with Pool(processes=batch) as pol:
            for idx in self.ts_chunk(batch=batch):
                fea = pol.map(new, self.ts_batch(table_series, idx))
                fea = pd.DataFrame.from_dict(fea)
                fea.index = list(idx)
                fea.to_sql(table_features, con, if_exists='append', dtype=None, index=True)
                con.commit()
                bar.update(1)
            pol.close()
            pol.join()

        bar.close()
        con.close()

