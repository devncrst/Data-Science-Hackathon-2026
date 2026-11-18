
import numpy as np, torch, torch.nn as nn, time, sys, scipy.sparse as sp
from scipy.spatial import cKDTree
sys.path.insert(0,'/home/participant-c1/claude_work'); import feats as FT
W='/home/participant-c1/claude_work/'
SEED=int(sys.argv[1]); DANN=float(sys.argv[2]); TAG=sys.argv[3]; NT=int(sys.argv[4])
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(NT)
def spatial_W(p, radius, sigma):
    tr=cKDTree(p); pairs=tr.query_pairs(radius,output_type='ndarray')
    dd=np.linalg.norm(p[pairs[:,0]]-p[pairs[:,1]],axis=1); w=np.exp(-dd**2/(2*sigma**2)); n=len(p)
    return sp.coo_matrix((np.r_[w,w],(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n)).tocsr()
def load(city):
    D=np.load(W+f'ts_{city}.npz'); c=FT.channels(FT.gapfill(D['arr']))
    p=np.c_[D['px'],D['py']].astype(float); return c,D['y'],p
cM,yM,pM=load('Madrid'); cA,yA,pA=load('Amsterdam')
mu=cM.reshape(-1,12).mean(0); sd=cM.reshape(-1,12).std(0)
def prep(c,p):
    z=(c-mu)/sd; N,T,C=z.shape; Wm=spatial_W(p,3,1.5); deg=np.asarray(Wm.sum(1)).ravel()
    s=((z.reshape(N,-1)+Wm@z.reshape(N,-1))/(1+deg[:,None])).reshape(N,T,C)
    return np.concatenate([z,s],-1).transpose(0,2,1).astype(np.float32)  # N,24,T
XM=prep(cM,pM); XA=prep(cA,pA)
class Net(nn.Module):
    def __init__(s,cin=24,h=64,e=64):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(cin,h,5,padding=2),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Conv1d(h,h,5,padding=4,dilation=2),nn.BatchNorm1d(h),nn.ReLU(),
            nn.Conv1d(h,h,3,padding=1),nn.BatchNorm1d(h),nn.ReLU(),nn.AdaptiveAvgPool1d(6),nn.Flatten(),
            nn.Dropout(0.3),nn.Linear(h*6,e),nn.ReLU())
        s.head=nn.Linear(e,4)
    def forward(s,x): return s.head(s.f(x))
rng=np.random.default_rng(SEED); idx=rng.permutation(len(yM)); va=idx[:7600]; tr=idx[7600:]
net=Net()
class GR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l): ctx.l=l; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.l*g, None
disc=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
opt=torch.optim.AdamW(list(net.parameters())+list(disc.parameters()),1e-3,weight_decay=1e-3)
XAt=torch.tensor(XA); bce=nn.BCEWithLogitsLoss()
cw=torch.tensor(len(yM)/(4*np.bincount(yM)[1:]),dtype=torch.float32)
lossf=nn.CrossEntropyLoss(weight=cw,label_smoothing=0.05)
Xt=torch.tensor(XM); yt=torch.tensor(yM-1)
from sklearn.metrics import f1_score
best=-1; t0=time.time(); EP=40
sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=EP*(len(tr)//512+1))
for ep in range(EP):
    net.train(); perm=rng.permutation(tr)
    for i in range(0,len(perm),512):
        b=perm[i:i+512]; xb=Xt[b]
        xb=xb+0.05*torch.randn_like(xb)  # light noise aug
        opt.zero_grad(); h=net.f(xb); l=lossf(net.head(h),yt[b])
        if DANN>0:
            ia=torch.randint(0,len(XAt),(len(b),)); ha=net.f(XAt[ia]+0.05*torch.randn_like(XAt[ia]))
            p=min(1.0,(ep*len(perm)+i)/(EP*len(perm)*0.3)); lam=DANN*p
            hh=GR.apply(torch.cat([h,ha]),lam); dl=bce(disc(hh).squeeze(1),torch.cat([torch.zeros(len(b)),torch.ones(len(b))]))
            l=l+dl
        l.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad(): pv=net(Xt[va]).argmax(1).numpy()+1; pa=net(torch.tensor(XA)).argmax(1).numpy()+1
    f=f1_score(yM[va],pv,average='macro'); fa=f1_score(yA,pa,average='macro')
    print(ep,round(f,4),'ams0shot',round(fa,4),round(time.time()-t0),flush=True)
    if f>best: best=f; torch.save(net.state_dict(),W+f'cnn_{TAG}.pt')
net.load_state_dict(torch.load(W+f'cnn_{TAG}.pt')); net.eval()
with torch.no_grad():
    EM=torch.cat([net.f(Xt[i:i+4096]) for i in range(0,len(Xt),4096)]).numpy()
    EA=net.f(torch.tensor(XA)).numpy()
np.savez(W+f'emb_{TAG}.npz',EM=EM,EA=EA,Wh=net.head.weight.detach().numpy(),bh=net.head.bias.detach().numpy(),va=va)
print('best val',best,flush=True)
