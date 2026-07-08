import numpy as np, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
 "font.family":"serif","mathtext.fontset":"stix","font.serif":["STIXTwoText","DejaVu Serif"],
 "font.size":11,"axes.linewidth":0.8,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.03,
})
CB=dict(blue="#0072B2",orange="#E69F00",green="#009E73",red="#D55E00",
        purple="#CC79A7",sky="#56B4E9",yellow="#F0E442",black="#222222",gray="#888888")

fig = plt.figure(figsize=(7.2,4.4))
gs = GridSpec(2, 5, height_ratios=[0.9,2.4], hspace=0.02, wspace=0.08,
              left=0.07, right=0.98, top=0.88, bottom=0.14)

# ---- five phase-space insets (Coulomb .. lambda* .. Dipole) ----
widths = [0.85, 0.62, 0.36, 0.62, 0.85]
for i, w in enumerate(widths):
    axe = fig.add_subplot(gs[0, i])
    axe.set_xlim(-1.5,1.5); axe.set_ylim(-1.5,1.5); axe.set_aspect("equal"); axe.axis("off")
    alpha = 0.30 if i==2 else 0.16
    axe.add_patch(Ellipse((0,0), width=2*w, height=1.1, facecolor=CB["blue"], alpha=alpha,
                           edgecolor=CB["blue"], linewidth=1.3))
    if i==0:
        axe.annotate("", xy=(1.35,0), xytext=(-1.35,0), clip_on=False,
                     arrowprops=dict(arrowstyle="-|>", color=CB["gray"], lw=0.8))
        axe.annotate("", xy=(0,1.35), xytext=(0,-1.35), clip_on=False,
                     arrowprops=dict(arrowstyle="-|>", color=CB["gray"], lw=0.8))
        axe.text(1.42, -0.10, "$x$", fontsize=9, color=CB["gray"], va="center", clip_on=False)
        axe.text(0.10, 1.40, "$p$", fontsize=9, color=CB["gray"], clip_on=False)

# ---- main gauge-orbit panel ----
ax = fig.add_subplot(gs[1, :])
lam = np.linspace(-1.25, 1.25, 400)
lam_star, Q, ymin = -0.2, 0.3, 0.05
y = ymin + Q*(lam-lam_star)**2
ax.plot(lam, y, color=CB["orange"], lw=2.1, zorder=3)

lam_pts = np.array([-1.0,-0.6,-0.2,0.3,1.0])
y_pts = ymin + Q*(lam_pts-lam_star)**2
for lp, yp in zip(lam_pts, y_pts):
    ax.plot([lp,lp],[yp, y.max()*1.14], color="#D8D8D8", lw=0.8, ls=(0,(2,3)), zorder=1)
ax.scatter([lam_star],[ymin], s=45, color=CB["orange"], edgecolor="#222222", linewidth=0.8, zorder=4)
ax.axhline(ymin, color=CB["gray"], lw=0.7, ls=(0,(4,3)), zorder=1)

ax.set_xlim(-1.3,1.3); ax.set_ylim(0, y.max()*1.38)
ax.set_yticks([])
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_position(("data",0))
ax.set_xticks([-1.0,-0.2,1.0]); ax.set_xticklabels(["Coulomb", r"$\lambda_*$", "Dipole"], fontsize=10.5)
ax.tick_params(axis="x", length=3)
ax.set_xlabel(r"gauge parameter $\lambda$", fontsize=12.5, labelpad=6)

ax.text(0.01, 0.97, r"$\langle a^\dagger a\rangle_\lambda$", transform=ax.transAxes,
        fontsize=12.5, color="#111111", va="top", ha="left")
ax.text(0.01, 0.80, r"$U_\lambda=\exp[\,i\lambda G(a+a^\dagger)\,]$:  $x\to x+2\lambda G$,  $p$ unchanged",
        transform=ax.transAxes, fontsize=10.5, color="#333333", va="top", ha="left")
ax.text(0.40, 0.55, r"$\langle a^\dagger a\rangle_\lambda=\langle a^\dagger a\rangle_0+C\lambda+Q\lambda^2$",
        transform=ax.transAxes, fontsize=12, color="#111111")
ax.text(lam_star+0.30, ymin + y.max()*0.16, "irreducible", ha="left", va="bottom",
        fontsize=9.5, color="#777777")

fig.suptitle("The gauge orbit of the photon dressing", fontsize=14, y=0.985)
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/00_gauge_orbit_overview.{ext}")
plt.close(fig); print("wrote 00_gauge_orbit_overview")
