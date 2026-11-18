
import numpy as np, pandas as pd, time
t0=time.time()
BANDS=['Blue','Green','Red','NIR','SWIR1','SWIR2']
EV={'Amsterdam':1945,'Madrid':1960}
D='/home/participant-c1/Building_Age_Classification/data/'
def prep(path, city):
    df=pd.read_parquet(path)
    cols=['px_key','py_key','year','weighted_mean_year','coverage']
    out=df[cols].copy()
    for b in BANDS:
        col=np.full(len(df),np.nan)
        for s in (1,2,3):
            v=df['qa_valid_%d'%s].fillna(False).values.astype(bool)
            col=np.where(np.isnan(col)&v, df['%s_%d'%(b,s)].astype(float).values, col)
        out[b]=col
    # also count of valid obs and median of valid obs per year
    nvalid=sum(df['qa_valid_%d'%s].fillna(False).astype(int).values for s in (1,2,3))
    out['nvalid']=nvalid
    out=out.dropna(subset=BANDS)
    out=out[out.Blue<=15000]
    out=out[(out[BANDS]>0).all(1)]
    pix=out.groupby(['px_key','py_key']).agg(wmy=('weighted_mean_year','first'),cov=('coverage','first')).reset_index()
    pix['pid']=np.arange(len(pix))
    out=out.merge(pix[['px_key','py_key','pid']],on=['px_key','py_key'])
    years=np.arange(1984,2026)
    T=len(years); N=len(pix)
    arr=np.full((N,T,6),np.nan,np.float32)
    yi=out.year.values-1984
    arr[out.pid.values,yi,:]=out[BANDS].values.astype(np.float32)
    obs=~np.isnan(arr[:,:,0])
    ev=EV[city]
    y=np.digitize(pix.wmy.values,[ev,1984,2004],right=True)+1
    np.savez_compressed('/home/participant-c1/claude_work/ts_%s.npz'%city, arr=arr, obs=obs, y=y, wmy=pix.wmy.values, cov=pix['cov'].values, px=pix.px_key.values, py=pix.py_key.values, years=years)
    print(city,N,'pixels', np.bincount(y), 'obs frac',obs.mean(), time.time()-t0, flush=True)
prep(D+'madrid_train.parquet','Madrid')
prep(D+'amsterdam_data.parquet','Amsterdam')
