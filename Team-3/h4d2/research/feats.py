
import numpy as np, pandas as pd
W='/home/participant-c1/claude_work/'
def gapfill(arr):
    N,T,C=arr.shape
    df=pd.DataFrame(arr.transpose(1,0,2).reshape(T,N*C))
    df=df.interpolate(axis=0,limit_direction='both')
    return df.values.reshape(T,N,C).transpose(1,0,2).astype(np.float32)
def channels(v):
    r=np.clip(v*2.75e-5-0.2,1e-3,None)
    B,G,R,N,S1,S2=[r[...,i] for i in range(6)]
    e=1e-6
    idx=[(N-R)/(N+R+e),(S1-N)/(S1+N+e),(S2-N)/(S2+N+e),(G-S1)/(G+S1+e),((S1+R)-(N+B))/((S1+R)+(N+B)+e)]
    bright=r.mean(-1)
    return np.concatenate([np.log(r),np.stack(idx,-1),np.log(bright)[...,None]],-1)  # 12 ch
CH=['lBlue','lGreen','lRed','lNIR','lSWIR1','lSWIR2','NDVI','NDBI','UI','MNDWI','BSI','lBright']
def city_norm(c, ref=None):
    # anomaly vs city-year median, scaled by city robust std per channel
    if ref is None:
        med=np.median(c,axis=0)              # T,C
        a=c-med
        mad=np.median(np.abs(a),axis=(0,1))*1.4826+1e-6
        ref=(med,mad)
    med,mad=ref
    return (c-med)/mad, ref
def feats(z, years):
    N,T,C=z.shape
    F={}; 
    tt=(years-years.mean())/years.std()
    dec=[(1984,1993),(1994,2003),(2004,2013),(2014,2025)]
    for j,(a,b) in enumerate(dec):
        m=(years>=a)&(years<=b); F.update({f'{CH[c]}_d{j}':z[:,m,c].mean(1) for c in range(C)})
    mean=z.mean(1)
    slope=(z*tt[None,:,None]).sum(1)/(tt**2).sum()
    resid=z-mean[:,None,:]-slope[:,None,:]*tt[None,:,None]
    e=years<=2003; l=years>=2004
    for c in range(C):
        F[f'{CH[c]}_mean']=mean[:,c]; F[f'{CH[c]}_slope']=slope[:,c]; F[f'{CH[c]}_rstd']=resid[:,:,c].std(1)
        F[f'{CH[c]}_estd']=z[:,e,c].std(1); F[f'{CH[c]}_lstd']=z[:,l,c].std(1)
    # changepoint: single mean shift, combined over channels
    cs=np.cumsum(z,axis=1); tot=cs[:,-1,:]
    best=np.full(N,-np.inf); bk=np.zeros(N,int); shift=np.zeros((N,C))
    var=resid.var(1)+1e-3
    for k in range(2,T-2):
        n1=k+1; n2=T-n1
        m1=cs[:,k,:]/n1; m2=(tot-cs[:,k,:])/n2
        d=m2-m1
        stat=((d**2)/var*(n1*n2/T)).sum(1)
        upd=stat>best; best[upd]=stat[upd]; bk[upd]=k; shift[upd]=d[upd]
    F['cp_year']=years[bk]+0.5; F['cp_stat']=np.log1p(best)
    for c in range(C): F[f'{CH[c]}_cpshift']=shift[:,c]
    # max jump of 3yr-smoothed series (brightness & NDVI & lSWIR1)
    for c in [6,4,11]:
        s=np.stack([z[:,max(0,t-1):t+2,c].mean(1) for t in range(T)],1)
        d=np.abs(np.diff(s,axis=1)); F[f'{CH[c]}_maxjump']=d.max(1); F[f'{CH[c]}_jumpyear']=years[1:][d.argmax(1)]
    return pd.DataFrame(F)
def build(city, ref=None):
    D=np.load(W+f'ts_{city}.npz')
    v=gapfill(D['arr']); c=channels(v); z,ref=city_norm(c,ref)
    df=feats(z,D['years'])
    return df, D, z, ref
if __name__=='__main__':
    import time; t=time.time()
    for city in ['Madrid','Amsterdam']:
        df,D,z,ref=build(city)
        df.to_parquet(W+f'feat_{city}.parquet'); np.save(W+f'z_{city}.npy',z.astype(np.float32))
        np.savez(W+f'ref_{city}.npz',med=ref[0],mad=ref[1])
        print(city,df.shape,time.time()-t,flush=True)
