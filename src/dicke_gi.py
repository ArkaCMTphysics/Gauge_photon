import numpy as np
from functools import reduce
from scipy.linalg import expm
sx=np.array([[0,1],[1,0]],float); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],float); I2=np.eye(2)
def kronN(m): return reduce(np.kron,m)
def op(o,l,Ns): return kronN([o if k==l else I2 for k in range(Ns)])
def bos(Nc): n=np.arange(Nc); a=np.diag(np.sqrt(n[1:]),1); return a,a.T

def dicke_GI(Ns,Nc,wc,wx,g,selfpol=True):
    """Garziano dipole-gauge GI Dicke (their Eq. 23), unrotated.
       coupling i g wc (a^dag-a) G ,  self-pol  g^2 wc G^2 ,  G=(1/sqrt Ns) sum sigma_x
       (collective coupling g = eta sqrt(Ns)). Zeeman wx/2 sum sigma_z."""
    a,ad=bos(Nc); Ib=np.eye(Nc); dimS=2**Ns; IS=np.eye(dimS)
    Jz=sum(op(sz,l,Ns) for l in range(Ns))/2.0
    G =sum(op(sx,l,Ns) for l in range(Ns))/np.sqrt(Ns)
    H = wc*np.kron(IS,ad@a) + wx*np.kron(Jz,Ib) \
        + 1j*g*wc*np.kron(G, ad-a)
    if selfpol:
        H = H + g**2*wc*np.kron(G@G, Ib)
    return H, a, ad, IS, G

def analyse(Ns,Nc,g,wc=1.0,wx=1.0,selfpol=True):
    H,a,ad,IS,G = dicke_GI(Ns,Nc,wc,wx,g,selfpol)
    E,V=np.linalg.eigh(H); psi=V[:,0]
    num=np.kron(IS,ad@a); Gop=np.kron(G,np.eye(a.shape[0]))
    p = np.kron(IS, 1j*(ad-a))          # p quadrature
    n0=float(np.real(psi.conj()@num@psi)); Q=float(np.real(psi.conj()@(Gop@Gop)@psi))
    C =float(np.real(psi.conj()@(-(Gop@p))@psi))   # cross term coefficient
    return n0,C,Q,(H,a,ad,IS,Gop,num,p,psi)

def photon_at(pack,alpha,g,wc=1.0):
    H,a,ad,IS,Gop,num,p,psi=pack
    # U_alpha = exp[ 2 i alpha eta (a+a^dag) Jx ] = exp[ i alpha g (a+a^dag) G ]  (since 2 eta Jx = g G)
    X=np.kron(IS,a+ad)
    U=expm(1j*alpha*g*(Gop@X))
    psl=U@psi
    return float(np.real(psl.conj()@num@psl))

print("GI Dicke (Garziano Eq.23). Check parabola & curvature, wc=wx=1, g=0.6")
for sp in (True,False):
    tag="WITH self-pol (gauge-invariant)" if sp else "NO self-pol (standard/violating)"
    print(f"\n--- {tag} ---")
    print(f"{'Ns':>3} {'<n>0':>8} {'C':>8} {'Q':>8}   parabola-check")
    for Ns in (1,2,3,4):
        Nc=28
        n0,C,Q,pack=analyse(Ns,Nc,0.6,selfpol=sp)
        errs=[abs(photon_at(pack,al,0.6)-(n0+C*al+Q*al**2)) for al in (-0.5,0.3,0.8)]
        print(f"{Ns:>3} {n0:8.4f} {C:8.4f} {Q:8.4f}   max|num-parab|={max(errs):.1e}")
