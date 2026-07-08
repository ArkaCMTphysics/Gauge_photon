import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from math import factorial
import os
mpl.use("Agg")
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------- publication style ----------------
mpl.rcParams.update({
 "font.family":"serif","mathtext.fontset":"stix","font.serif":["STIXTwoText","DejaVu Serif"],
 "font.size":11,"axes.labelsize":12,"axes.titlesize":12,"legend.fontsize":9.5,
 "xtick.labelsize":10,"ytick.labelsize":10,"axes.linewidth":0.8,
 "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
 "xtick.major.size":4,"ytick.major.size":4,"xtick.minor.size":2.2,"ytick.minor.size":2.2,
 "legend.frameon":False,"axes.grid":False,"figure.dpi":140,"savefig.dpi":300,
 "savefig.bbox":"tight","savefig.pad_inches":0.02,"lines.linewidth":1.6,
})
# Okabe-Ito colourblind-safe palette
CB=dict(blue="#0072B2",orange="#E69F00",green="#009E73",red="#D55E00",
        purple="#CC79A7",sky="#56B4E9",yellow="#F0E442",black="#222222")
def save(fig,name):
    for ext in ("pdf","png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig); print("wrote",name)

# ---------------- QRM exact diagonalisation ----------------
def ops(Nf):
    n=np.arange(Nf); a=np.diag(np.sqrt(n[1:]),1); ad=a.T
    return a,ad
def qrm(g,Delta,w=1.0,Nf=160,parity=None):
    a,ad=ops(Nf); I2=np.eye(2)
    sx=np.array([[0.,1],[1,0]]); sz=np.array([[1.,0],[0,-1]])
    H=np.kron(w*ad@a,I2)+np.kron(g*(a+ad),sx)+np.kron(np.eye(Nf),Delta*sz)
    return H,a,ad,Nf
def jc(g,Delta,w=1.0,Nf=160):
    a,ad=ops(Nf)
    sp=np.array([[0.,1],[0,0]]); sm=sp.T; sz=np.array([[1.,0],[0,-1]])
    H=np.kron(w*ad@a,np.eye(2))+g*(np.kron(a,sp)+np.kron(ad,sm))+np.kron(np.eye(Nf),Delta*sz)
    return H
def gs_obs(g,Delta,w=1.0,Nf=160):
    H,a,ad,Nf=qrm(g,Delta,w,Nf); E,V=np.linalg.eigh(H); psi=V[:,0]
    num=np.kron(ad@a,np.eye(2)); sx=np.kron(np.eye(Nf),np.array([[0.,1],[1,0]])); x=np.kron(a+ad,np.eye(2))
    return E[0], float(psi@num@psi), float(psi@(sx@x)@psi), E

def HC_corrected(g,Delta,w=1.0,Nf=160):
    """eq:corrected-Coulomb = H^(alpha=0), the genuine gauge family's Coulomb endpoint
    (D=0, so wc_tilde=wc and eta=g/wc); NOT the naive/uncorrected H_C used by gs_obs above,
    which Sec. 3.2 states is not part of the genuine gauge family."""
    from scipy.linalg import cosm, sinm
    a,ad=ops(Nf); I2=np.eye(2)
    sy=np.array([[0.,-1j],[1j,0]]); sz=np.array([[1.,0],[0,-1]],dtype=complex)
    eta=g/w; x=a+ad
    H=w*np.kron(ad@a,I2)+Delta*(np.kron(cosm(2*eta*x),sz)+np.kron(sinm(2*eta*x),sy))
    return H,a,ad,Nf

def gs_obs_genuine(g,Delta,w=1.0,Nf=160):
    """Ground-state moments (n0,C) of the genuine alpha-gauge family at alpha=0 (Coulomb),
    using the universal C = i<sigma_x(ad-a)>_0 convention of eq:universal-parabola (Q=1 exactly,
    since G=sigma_x, G^2=I). Because U_alpha = U_lambda|_{lambda=-alpha*eta} is exactly the
    universal displacement operator, the parabola n(alpha) = n0 - C*eta*alpha + eta^2*alpha^2
    follows in closed form from this single diagonalisation -- no further sweep is needed."""
    H,a,ad,Nf=HC_corrected(g,Delta,w,Nf); E,V=np.linalg.eigh(H); psi=V[:,0]
    I2=np.eye(2)
    num=np.kron(ad@a,I2)
    Gop=np.kron(np.eye(Nf),np.array([[0.,1],[1,0]],dtype=complex))  # sigma_x on matter
    padag_a=np.kron(ad-a,I2)
    n0=float(np.real(psi.conj()@num@psi))
    C=float(np.real(1j*(psi.conj()@(Gop@padag_a)@psi)))
    return n0,C
def parity_spectrum(Hfun,g,Delta,nlev,w=1.0,Nf=120):
    # parity P = exp(i pi (a^dag a + (1-sx)/2)) ; build and split
    H,*_=(Hfun(g,Delta,w,Nf) if Hfun is qrm else (jc(g,Delta,w,Nf),))
    if Hfun is qrm: H=H
    a,ad=ops(Nf)
    nph=np.kron(np.diag(np.arange(Nf)),np.eye(2))
    sx=np.kron(np.eye(Nf),np.array([[0.,1],[1,0]]))
    Ppar=np.kron(np.diag((-1.0)**np.arange(Nf)),np.array([[0.,1],[1,0]]))  # parity op (sx flavour)
    E,V=np.linalg.eigh(H)
    par=np.array([np.sign(np.real(V[:,k]@Ppar@V[:,k])) if abs(V[:,k]@Ppar@V[:,k])>1e-6 else 0 for k in range(2*Nf)])
    return E,par

# ================= FIG: energy spectrum Rabi vs JC =================
def fig_spectrum():
    gs=np.linspace(0,1.0,200); Delta=0.7; nlev=12; Nf=90
    Rab=np.zeros((len(gs),nlev)); JC=np.zeros((len(gs),nlev))
    for i,g in enumerate(gs):
        H,*_=qrm(g,Delta,1.0,Nf); Rab[i]=np.linalg.eigvalsh(H)[:nlev]
        JC[i]=np.linalg.eigvalsh(jc(g,Delta,1.0,Nf))[:nlev]
    fig,ax=plt.subplots(figsize=(5.4,4.0))
    for k in range(nlev):
        ax.plot(gs,Rab[:,k],color=CB["blue"],lw=1.3,zorder=3)
        ax.plot(gs,JC[:,k],color=CB["red"],lw=1.0,ls=(0,(4,2)),zorder=2)
    ax.plot([],[],color=CB["blue"],lw=1.3,label="quantum Rabi")
    ax.plot([],[],color=CB["red"],lw=1.0,ls=(0,(4,2)),label="Jaynes--Cummings")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$E_k/\omega_c$")
    ax.set_xlim(0,1.0); ax.set_ylim(-1.6,6.0); ax.legend(loc="upper left")
    ax.set_title(r"Low-lying spectrum, $\Delta=0.7\,\omega_c$")
    save(fig,"01_spectrum_rabi_jc")

# ================= FIG: GS photon number Rabi vs JC =================
def fig_gs_photon():
    gs=np.linspace(0,1.0,140); Delta=0.7; Nf=160
    nR=[]; nJ=[]
    for g in gs:
        _,n0,_,_=gs_obs(g,Delta,1.0,Nf); nR.append(n0)
        Hj=jc(g,Delta,1.0,Nf); E,V=np.linalg.eigh(Hj); psi=V[:,0]
        a,ad=ops(Nf); num=np.kron(ad@a,np.eye(2)); nJ.append(float(psi@num@psi))
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    ax.plot(gs,nR,color=CB["blue"],label="quantum Rabi")
    ax.plot(gs,nJ,color=CB["red"],ls=(0,(4,2)),label="Jaynes--Cummings")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\rm gs}$")
    ax.set_xlim(0,1.0); ax.set_ylim(bottom=-0.01); ax.legend(loc="upper left")
    ax.set_title(r"Ground-state photon number, $\Delta=0.7\,\omega_c$")
    save(fig,"02_gs_photon_rabi_jc")

# ================= FIG: discriminant & min photon vs g (genuine alpha-gauge family) =================
def fig_discriminant_min():
    gs=np.linspace(0.02,0.7,120); Deltas=[0.3,0.5,0.7]; cols=[CB["green"],CB["orange"],CB["blue"]]
    figD,axD=plt.subplots(figsize=(5.0,3.7)); figM,axM=plt.subplots(figsize=(5.0,3.7))
    for D,c in zip(Deltas,cols):
        Dv=[]; Mv=[]
        for g in gs:
            n0,C=gs_obs_genuine(g,D); Dv.append(C**2-4*n0); Mv.append(n0-C**2/4)
        axD.plot(gs,Dv,color=c,label=fr"$\Delta={D}\,\omega_c$")
        axM.plot(gs,Mv,color=c,label=fr"$\Delta={D}\,\omega_c$")
    axD.axhline(0,color=CB["black"],lw=0.7,ls=":")
    axD.set_xlabel(r"coupling $g/\omega_c$"); axD.set_ylabel(r"$\mathcal{D}=C^{2}-4\langle a^{\dagger}a\rangle_{0}$")
    axD.set_xlim(0,0.7); axD.legend(loc="lower left")
    axD.set_title(r"Discriminant $\mathcal{D}<0$: no photon-empty gauge")
    save(figD,"03_discriminant_vs_g")
    axM.set_xlabel(r"coupling $g/\omega_c$"); axM.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\min}=\langle a^{\dagger}a\rangle_{0}-C^{2}/4$")
    axM.set_xlim(0,0.7); axM.set_ylim(bottom=0); axM.legend(loc="upper left")
    axM.set_title(r"Invariant minimal photon content")
    save(figM,"04_min_photon_vs_g")

# ================= FIG: photon number across gauges (genuine alpha-gauge family) =================
def fig_gauge_family():
    gs=np.linspace(0.0,0.8,140); Delta=0.7
    alphas=[0.0,0.25,0.5,0.75,1.0]; cols=plt.cm.viridis(np.linspace(0.1,0.85,len(alphas)))
    n0s=[]; Cs=[]
    for g in gs:
        n0,C=gs_obs_genuine(g,Delta); n0s.append(n0); Cs.append(C)
    n0s=np.array(n0s); Cs=np.array(Cs); etas=gs
    fig,ax=plt.subplots(figsize=(5.2,3.8))
    for al,c in zip(alphas,cols):
        n_al=n0s-Cs*etas*al+(etas*al)**2
        ax.plot(gs,n_al,color=c,label=fr"$\alpha={al:.2f}$")
    ax.plot(gs,n0s-Cs**2/4,color=CB["black"],lw=1.4,ls=(0,(1,1)),label=r"minimum (invariant)")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\alpha}$")
    ax.set_xlim(0,0.8); ax.set_ylim(bottom=0); ax.legend(loc="upper left",ncol=2)
    ax.set_title(r"Photon number across gauges, $\Delta=0.7\,\omega_c$")
    save(fig,"05_photon_across_gauges")

# ================= FIG: n vs alpha parabolas (genuine alpha-gauge family) =================
def fig_n_vs_kappa():
    Delta=0.7; ggs=[0.2,0.4,0.6]; cols=[CB["green"],CB["orange"],CB["blue"]]
    als=np.linspace(-0.3,1.5,400)
    fig,ax=plt.subplots(figsize=(5.0,3.7))
    for g,c in zip(ggs,cols):
        n0,C=gs_obs_genuine(g,Delta); eta=g
        ax.plot(als,n0-C*eta*als+(eta*als)**2,color=c,label=fr"$g={g}\,\omega_c$")
        amin=C/(2*eta); ax.scatter([amin],[n0-C**2/4],color=c,s=22,zorder=5,edgecolor="k",linewidth=0.4)
    ax.axvline(0,color=CB["black"],lw=0.7,ls=":"); ax.axvline(1,color=CB["black"],lw=0.7,ls=":")
    ax.set_xlabel(r"gauge parameter $\alpha$ (Coulomb$=0$, dipole$=1$)"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\alpha}$")
    ax.set_xlim(-0.3,1.5); ax.set_ylim(bottom=0); ax.legend(loc="upper center")
    ax.set_title(r"Unit-curvature gauge orbit (dots: invariant minima)")
    save(fig,"06_photon_vs_kappa")

# ================= tunable coherent-state recursion (CORRECT variant) =================
def tunable_c(alpha,g,Delta,Pi,Ntr):
    c=np.zeros(Ntr+2); c[0]=1.0
    j=np.arange(Ntr+2); coef=(2*alpha)**j/np.array([factorial(k) for k in j])
    for nn in range(1,Ntr+1):
        s=sum(((-1)**nn)*coef[jj]*c[nn-jj] for jj in range(nn+1))
        c[nn+1]=(-(nn+Pi*Delta)*c[nn]-(alpha+g)*c[nn-1])/(g*(nn+1))+Pi*Delta*s/(g*(nn+1))
    return c
def root_alpha(g,Delta,Pi,Ntr,br=(-1.5,1.2),Np=3001):
    xs=np.linspace(*br,Np); f=np.array([tunable_c(x,g,Delta,Pi,Ntr)[Ntr+1] for x in xs]); rts=[]
    for i in range(len(xs)-1):
        if f[i]*f[i+1]<0:
            a0,a1=xs[i],xs[i+1]
            for _ in range(70):
                am=.5*(a0+a1); fm=tunable_c(am,g,Delta,Pi,Ntr)[Ntr+1]
                a1,a0=(am,a0) if f[i]*fm<=0 else (a1,am)
            rts.append(.5*(a0+a1))
    return np.array(rts)

def fig_cn():
    """Coefficient decay & convergence, computed at high precision (mpmath).
    For each truncation N the recursion is faithful only up to n=N (beyond the
    truncation the dominant solution takes over), so each series is plotted only
    on n<=N. Increasing N extends the SAME envelope to smaller |c_n|."""
    import mpmath as mp
    mp.mp.dps=80
    g,Delta,Pi=mp.mpf("0.8"),mp.mpf("0.7"),1
    def cseq(al,Ntr):
        al=mp.mpf(al); c=[mp.mpf(0)]*(Ntr+2); c[0]=mp.mpf(1)
        cf=[(2*al)**j/mp.factorial(j) for j in range(Ntr+2)]
        for n in range(1,Ntr+1):
            ss=mp.fsum(((-1)**n)*cf[j]*c[n-j] for j in range(n+1))
            c[n+1]=(-(n+Pi*Delta)*c[n]-(al+g)*c[n-1])/(g*(n+1))+Pi*Delta*ss/(g*(n+1))
        return c
    def aN(N,guess=-0.395748): return mp.findroot(lambda a:cseq(a,N)[N+1], mp.mpf(guess))

    # converged envelope (large N)
    Nenv=42; cE=cseq(aN(Nenv),Nenv)
    nE=np.arange(2,Nenv-4); yE=np.array([float(abs(cE[n])) for n in nE])

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5))
    ax=axes[0]
    ax.plot(nE,yE,color=CB["black"],lw=1.2,zorder=1,label=r"converged ($N=42$)")
    trunc={8:("o",CB["green"]),12:("s",CB["orange"]),16:("D",CB["blue"]),24:("^",CB["purple"])}
    for N,(m,col) in trunc.items():
        c=cseq(aN(N),N); ns=np.arange(2,N+1)              # only up to the truncation
        ys=np.array([float(abs(c[n])) for n in ns])
        ax.scatter(ns,ys,marker=m,s=26,facecolor="none",edgecolor=col,linewidth=1.1,
                   zorder=3,label=fr"$N={N}$")
    ax.set_yscale("log"); ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$|c_n|$")
    ax.set_xlim(0,30); ax.set_ylim(1e-26,3); ax.legend(loc="upper right",fontsize=8.2)
    ax.set_title(r"Convergence of the coefficients")

    # factorial-law panel: ln|c_n| vs ln(n!)  -> slope is the factorial exponent
    ax2=axes[1]
    xnf=np.array([float(mp.log(mp.factorial(int(n)))) for n in nE])
    yln=np.log(yE)
    ax2.plot(xnf, yln, color=CB["red"], marker="o", ms=3.5, lw=0, zorder=3, label=r"$\ln|c_n|$")
    m=(nE>=6)                                   # clean asymptotic window
    sl,ic=np.polyfit(xnf[m], yln[m],1)
    xx=np.array([xnf[2],xnf[-1]])
    ax2.plot(xx, sl*xx+ic, color=CB["blue"], ls=(0,(5,2)), zorder=2,
             label=fr"slope $={sl:.2f}$")
    ax2.set_xlabel(r"$\ln\,n!$")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel(r"$\ln|c_n|$", rotation=270, labelpad=14)
    ax2.legend(loc="lower left")
    ax2.set_title(r"$|c_n|\sim (n!)^{%.2f}$: super-exponential"%sl)
    save(fig,"07_cn_decay")

def fig_alpha_conv():
    g,Delta,Pi=0.8,0.7,1; Ns=range(4,16); Egs=-1.0166; vals=[]
    for N in Ns:
        al=root_alpha(g,Delta,Pi,N); vals.append(al[np.argmin(np.abs(al*g-Pi*Delta-Egs))])
    fig,ax=plt.subplots(figsize=(5.0,3.5))
    ax.plot(list(Ns),vals,marker="o",color=CB["red"],ms=6,lw=1.3)
    ax.axhline(vals[-1],color=CB["blue"],lw=0.8,ls=":")
    ax.text(0.5,0.18,fr"$\alpha_\infty\approx{vals[-1]:.4f}$",transform=ax.transAxes,color=CB["blue"])
    ax.set_xlabel(r"truncation $N$"); ax.set_ylabel(r"root $\alpha$ of $c_{N+1}(\alpha)=0$")
    ax.set_title(r"Convergence of the coherent-state displacement",pad=8)
    save(fig,"08_alpha_convergence")

# ================= excited-state photon spectrum =================
def fig_excited():
    gs=np.linspace(0.01,1.0,160); Delta=0.7; Nf=160; nshow=6
    a,ad=ops(Nf); num=np.kron(ad@a,np.eye(2))
    Ppar=np.kron(np.diag((-1.0)**np.arange(Nf)),np.array([[0.,1],[1,0]]))
    data=[]
    for g in gs:
        H,*_=qrm(g,Delta,1.0,Nf); E,V=np.linalg.eigh(H)
        row=[]
        for k in range(nshow):
            psi=V[:,k]; row.append((float(psi@num@psi), np.real(psi@Ppar@psi)))
        data.append(row)
    data=np.array(data)
    fig,ax=plt.subplots(figsize=(5.4,4.0))
    for k in range(nshow):
        nk=data[:,k,0]; par=data[:,k,1]
        ev=par>=0
        ax.scatter(gs[ev],nk[ev],s=7,color=CB["blue"])
        ax.scatter(gs[~ev],nk[~ev],s=7,color=CB["red"])
    ax.scatter([],[],s=12,color=CB["blue"],label="even parity")
    ax.scatter([],[],s=12,color=CB["red"],label="odd parity")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_k$")
    ax.set_xlim(0,1.0); ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    ax.set_title(r"Photon content of the lowest eigenstates, $\Delta=0.7\,\omega_c$")
    save(fig,"09_excited_photon_spectrum")

# ================= timing (their fit + measured points, restyled) =================
def fig_timing():
    # reported fit:  log10? -> their fit log(T) ~ 0.319 log(N!) - 0.950 (natural log)
    Nmeas=np.array([6,7,8,9,10,11]); logT=np.array([1.15,1.75,2.45,3.12,3.87,4.63])  # read from plot
    Nall=np.arange(6,20); lnfac=np.array([np.log(float(factorial(n))) for n in Nall])
    fit=0.319*lnfac-0.950
    fig,ax=plt.subplots(figsize=(5.0,3.6))
    ax.plot(Nall, np.exp(fit), color=CB["blue"], lw=1.4, label=r"fit $\propto (N!)^{0.319}$")
    ax.scatter(Nmeas, np.exp(logT), color=CB["red"], s=24, zorder=5, label="measured")
    ax.set_yscale("log"); ax.set_xlabel(r"truncation $N$"); ax.set_ylabel(r"wall-clock time (s)")
    ax.legend(loc="upper left"); ax.set_title(r"Cost of the excited-state evaluation")
    save(fig,"10_timing_loglinear")
    fig,ax=plt.subplots(figsize=(5.0,3.6))
    ax.scatter(Nmeas,logT,color=CB["red"],s=26,zorder=5,label=r"$\log T$ (measured)")
    ax.plot(Nmeas,0.319*np.array([np.log(float(factorial(n))) for n in Nmeas])-0.950,
            color=CB["green"],ls=(0,(5,2)),label=r"$0.319\,\log N!-0.950$")
    ax.set_xlabel(r"truncation $N$"); ax.set_ylabel(r"$\log T$")
    ax.legend(loc="upper left"); ax.set_title(r"Factorial scaling, $\log T \propto \log N!$")
    save(fig,"11_timing_logfit")

def _run_all():
    for f in (fig_spectrum,fig_gs_photon,fig_discriminant_min,fig_gauge_family,fig_n_vs_kappa,
              fig_cn,fig_alpha_conv,fig_excited,fig_timing,fig_alpha_min,fig_ed_photon,fig_ed_tunable):
        f()
    print("ALL DONE")

# ---- direct Fock construction of the tunable ground state (validated vs ED) ----
from math import lgamma as _lg, log as _log
def _tunable_state_fock(al,c,Ntr,Nf=120):
    v=np.zeros(Nf); sgn=-1.0 if al<0 else 1.0; la=_log(abs(al)) if al!=0 else -1e9
    for n in range(Ntr+1):
        if c[n]==0: continue
        sc=np.sign(c[n]); lc=_log(abs(c[n]))
        for m in range(Nf-n):
            lt=lc+m*la-0.5*_lg(m+1)+0.5*(_lg(m+n+1)-_lg(m+1))
            v[m+n]+=sc*(sgn**m)*np.exp(lt)
    return v/np.linalg.norm(v)
def tunable_photon(g,Delta,Pi,Ntr,Nf=120):
    al=root_alpha(g,Delta,Pi,Ntr); Egs=gs_obs(g,Delta)[0]
    al=al[np.argmin(np.abs(al*g-Pi*Delta-Egs))]
    v=_tunable_state_fock(al,tunable_c(al,g,Delta,Pi,Ntr),Ntr,Nf)
    return float(np.sum(np.arange(Nf)*v**2))

# ============== FIG 12: alpha_min(g) ==============
def fig_alpha_min():
    gs=np.linspace(0.05,0.7,90); Deltas=[0.3,0.5,0.7]; cols=[CB["green"],CB["orange"],CB["blue"]]
    fig,ax=plt.subplots(figsize=(5.0,3.7))
    for D,c in zip(Deltas,cols):
        am=[]
        for g in gs:
            n0,C=gs_obs_genuine(g,D); eta=g              # eta = g_D/omega_tilde (D=0 => omega_tilde=1)
            am.append(C/(2*eta))                          # alpha_min = C/(2 eta), genuine family
        ax.plot(gs,am,color=c,label=fr"$\Delta={D}\,\omega_c$")
    ax.axhline(0,color=CB["black"],lw=0.6,ls=":"); ax.axhline(1,color=CB["black"],lw=0.6,ls=":")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\alpha_{\min}$")
    ax.set_xlim(0,0.7); ax.set_ylim(0,1); ax.legend(loc="upper left")
    ax.set_title(r"Gauge of minimal photon content $\alpha_{\min}=C/2\eta$")
    save(fig,"12_alpha_min_vs_g")

# ============== FIG 13: ED ground-state photon number ==============
def fig_ed_photon():
    gs=np.linspace(0.0,1.0,140); Deltas=[0.3,0.5,0.7]; cols=[CB["green"],CB["orange"],CB["blue"]]
    fig,ax=plt.subplots(figsize=(5.0,3.7))
    for D,c in zip(Deltas,cols):
        ax.plot(gs,[gs_obs(g,D)[1] for g in gs],color=c,label=fr"$\Delta={D}\,\omega_c$")
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{0}$")
    ax.set_xlim(0,1.0); ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    ax.set_title(r"Ground-state photon number (exact diagonalisation)")
    save(fig,"13_ed_photon")

# ============== FIG 14: ED vs tunable coherent state ==============
def fig_ed_tunable():
    gline=np.linspace(0.02,0.95,120); gmk=np.linspace(0.2,0.9,15); Delta=0.7
    fig,ax=plt.subplots(figsize=(5.0,3.7))
    ax.plot(gline,[gs_obs(g,Delta)[1] for g in gline],color=CB["blue"],lw=1.6,label="exact diagonalisation",zorder=2)
    ax.scatter(gmk,[tunable_photon(g,Delta,1,14) for g in gmk],s=26,facecolor="none",
               edgecolor=CB["red"],linewidth=1.2,label=r"tunable coherent state ($N=14$)",zorder=3)
    ax.set_xlabel(r"coupling $g/\omega_c$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{0}$")
    ax.set_xlim(0,1.0); ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    ax.set_title(r"Validation: analytic vs. numerical, $\Delta=0.7\,\omega_c$")
    save(fig,"14_ed_vs_tunable")

if __name__=="__main__":
    _run_all()
