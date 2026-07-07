import numpy as np, os
import makefigs as M           # style + palette
import dicke_gi as G
from scipy.linalg import expm
plt=M.plt; CB=M.CB
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT,exist_ok=True)

def moments(Ns,g,Nc,selfpol=True,wc=1.0,wx=1.0):
    H,a,ad,IS,Gm=G.dicke_GI(Ns,Nc,wc,wx,g,selfpol)
    E,V=np.linalg.eigh(H); psi=V[:,0]
    num=np.kron(IS,ad@a); Gop=np.kron(Gm,np.eye(Nc)); p=np.kron(IS,1j*(ad-a))
    n0=float(np.real(psi.conj()@num@psi)); Q=float(np.real(psi.conj()@(Gop@Gop)@psi))
    Cl=float(np.real(psi.conj()@(Gop@p)@psi))
    return n0,Cl,Q
def ncut(Ns,g): return int(max(24, 20+34*g))

cols=[CB["blue"],CB["orange"],CB["green"],CB["purple"]]

# Mean-field (thermodynamic-limit) superradiant critical coupling of the standard
# (uncorrected, D=0) Dicke Hamiltonian, from Holstein-Primakoff linearization +
# Bogoliubov stability analysis of H = wc a^dag a + wx b^dag b
#   + i*g*wc/sqrt(Ns) * (a^dag - a) * sqrt(Ns/2)*(b+b^dag);
# instability threshold at g_c = (1/2) sqrt(wx/wc).
def gc_meanfield(wc=1.0, wx=1.0): return 0.5*np.sqrt(wx/wc)

# ---- Fig 15: GI parabolas ----
def fig_parabolas():
    g=0.8; lam=np.linspace(-1.2,1.2,400)
    fig,ax=plt.subplots(figsize=(5.4,4.0))
    for Ns,c in zip((1,2,3,4),cols):
        n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=True)
        ax.plot(lam,n0+Cl*lam+Q*lam**2,color=c,label=fr"$N_s={Ns}$ ($Q={Q:.2f}$)")
        lm=-Cl/(2*Q); ax.scatter([lm],[n0-Cl**2/(4*Q)],color=c,s=22,zorder=5,edgecolor="k",linewidth=0.4)
    ax.set_xlabel(r"gauge parameter $\lambda$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\lambda}$")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(bottom=0); ax.legend(loc="upper center",ncol=2,fontsize=8.5)
    ax.set_title(r"Gauge-invariant Dicke ($g=0.8$): curvature $Q\!\approx\!1$, regulated")
    M.save(fig,"15_dicke_gauge_parabolas")

# ---- Fig 16: curvature GI vs standard (the headline) ----
def fig_curvature():
    gs=np.linspace(0.05,1.0,26)
    fig,ax=plt.subplots(figsize=(5.6,4.1))
    for Ns,c in zip((1,2,3,4),cols):
        Qg=[moments(Ns,g,ncut(Ns,g),selfpol=True)[2]  for g in gs]
        Qs=[moments(Ns,g,ncut(Ns,g),selfpol=False)[2] for g in gs]
        ax.plot(gs,Qg,color=c,lw=1.8,label=fr"$N_s={Ns}$")
        ax.plot(gs,Qs,color=c,lw=1.1,ls=(0,(4,2)),alpha=0.9)
    ax.axhline(1.0,color=CB["black"],lw=0.7,ls=":")
    ax.plot([],[],color=CB["black"],lw=1.8,label="gauge-invariant (TRK)")
    ax.plot([],[],color=CB["black"],lw=1.1,ls=(0,(4,2)),label="standard (no diamagn.)")
    ax.set_xlabel(r"coupling $g$"); ax.set_ylabel(r"curvature $Q=\langle G^{2}\rangle$")
    ax.set_xlim(0,1.0); ax.legend(loc="upper left",ncol=2,fontsize=8.2)
    ax.set_title(r"Diamagnetic term regulates the curvature")
    M.save(fig,"16_dicke_curvature_vs_g")

# ---- Fig 17: discriminant GI ----
def fig_discriminant():
    gs=np.linspace(0.05,1.0,26)
    fig,ax=plt.subplots(figsize=(5.2,3.9))
    for Ns,c in zip((1,2,3,4),cols):
        Dv=[]
        for g in gs:
            n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=True); Dv.append(Cl**2-4*Q*n0)
        ax.plot(gs,Dv,color=c,marker="o",ms=3,lw=1.4,label=fr"$N_s={Ns}$")
    ax.axhline(0,color=CB["black"],lw=0.7,ls=":")
    ax.set_xlabel(r"coupling $g$"); ax.set_ylabel(r"$\mathcal{D}=C^{2}-4Q\langle a^{\dagger}a\rangle_{0}$")
    ax.set_xlim(0,1.0); ax.legend(loc="lower left",ncol=2,fontsize=8.5)
    ax.set_title(r"$\mathcal{D}<0$ (gauge-invariant Dicke): minimum stays positive")
    M.save(fig,"17_dicke_discriminant")

# ---- Fig 18: gauge of minimal photon content vs g (GI vs standard) ----
def fig_alpha_min():
    gs=np.linspace(0.05,1.0,26); gc=gc_meanfield()
    fig,ax=plt.subplots(figsize=(5.6,4.1))
    for Ns,c in zip((1,2,3,4),cols):
        amG=[]; amS=[]
        for g in gs:
            n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=True);  amG.append(-Cl/(2*Q*g))
            n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=False); amS.append(-Cl/(2*Q*g))
        ax.plot(gs,amG,color=c,lw=1.8,label=fr"$N_s={Ns}$")
        ax.plot(gs,amS,color=c,lw=1.1,ls=(0,(4,2)),alpha=0.9)
    ax.axvline(gc,color=CB["black"],lw=0.9,ls=":")
    ax.text(gc,ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 0,r"$g_c$",ha="center",va="bottom",fontsize=9)
    ax.plot([],[],color=CB["black"],lw=1.8,label="gauge-invariant (TRK)")
    ax.plot([],[],color=CB["black"],lw=1.1,ls=(0,(4,2)),label="standard (no diamagn.)")
    ax.set_xlabel(r"coupling $g$"); ax.set_ylabel(r"gauge of minimal photon content $\alpha_{\min}$")
    ax.set_xlim(0,1.0); ax.legend(loc="best",ncol=2,fontsize=8.0)
    ax.set_title(r"Minimal-photon gauge vs coupling")
    M.save(fig,"18_dicke_alpha_min_vs_g")

# ---- Fig 19: invariant minimal photon number vs g (GI vs standard) ----
def fig_min_photon():
    gs=np.linspace(0.05,1.0,26); gc=gc_meanfield()
    fig,ax=plt.subplots(figsize=(5.6,4.1))
    for Ns,c in zip((1,2,3,4),cols):
        nmG=[]; nmS=[]
        for g in gs:
            n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=True);  nmG.append(n0-Cl**2/(4*Q))
            n0,Cl,Q=moments(Ns,g,ncut(Ns,g),selfpol=False); nmS.append(n0-Cl**2/(4*Q))
        ax.plot(gs,nmG,color=c,lw=1.8,label=fr"$N_s={Ns}$")
        ax.plot(gs,nmS,color=c,lw=1.1,ls=(0,(4,2)),alpha=0.9)
    ax.axvline(gc,color=CB["black"],lw=0.9,ls=":")
    ax.text(gc,ax.get_ylim()[1],r"$g_c$",ha="center",va="bottom",fontsize=9)
    ax.plot([],[],color=CB["black"],lw=1.8,label="gauge-invariant (TRK)")
    ax.plot([],[],color=CB["black"],lw=1.1,ls=(0,(4,2)),label="standard (no diamagn.)")
    ax.set_xlabel(r"coupling $g$"); ax.set_ylabel(r"$\langle a^{\dagger}a\rangle_{\min}$")
    ax.set_xlim(0,1.0); ax.set_ylim(bottom=0); ax.legend(loc="upper left",ncol=2,fontsize=8.0)
    ax.set_title(r"Invariant minimal photon number vs coupling")
    M.save(fig,"19_dicke_min_photon_vs_g")

# convergence guard
print("Ncut convergence (Ns=4, GI):")
for g in (0.6,1.0):
    for nc in (ncut(4,g), ncut(4,g)+14):
        print(f"  g={g} Nc={nc}:", np.round(moments(4,g,nc,True),5))
for f in (fig_parabolas,fig_curvature,fig_discriminant,fig_alpha_min,fig_min_photon): f()
print("DONE")
