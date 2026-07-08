import numpy as np, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Circle
from matplotlib.gridspec import GridSpec

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
 "font.family":"serif","mathtext.fontset":"stix","font.serif":["STIXTwoText","DejaVu Serif"],
 "font.size":11,"axes.linewidth":0.9,"savefig.dpi":450,"savefig.bbox":"tight","savefig.pad_inches":0.03,
})
CB=dict(blue="#0072B2",orange="#E69F00",green="#009E73",red="#D55E00",
        purple="#CC79A7",sky="#56B4E9",yellow="#F0E442",black="#222222",gray="#888888")
blue_rgb = mcolors.to_rgb(CB["blue"])

fig = plt.figure(figsize=(7.2,4.4))
gs = GridSpec(2, 5, height_ratios=[0.9,2.4], hspace=0.02, wspace=0.08,
              left=0.09, right=0.98, top=0.88, bottom=0.14)

# ---- shared gauge-orbit data: <n>_lambda = ymin + Q*(lambda-lambda*)^2 (schematic).
# <p^2>_lambda obeys the SAME parabola shape, since <p^2>_lambda - <p^2>_min = 4*(<n>_lambda-<n>_min)
# exactly (from n=(x^2+p^2-2)/4 and x^2 being lambda-invariant) -- so the insets' height below
# reuses y_pts directly, tying the "squeeze" width to the same curve plotted in the main panel. ----
lam_star, Q, ymin = -0.2, 0.3, 0.05
lam_pts = np.array([-1.0, -0.6, -0.2, 0.3, 1.0])
y_pts = ymin + Q*(lam_pts-lam_star)**2

# ---- five phase-space insets (Coulomb .. lambda* .. Dipole): U_lambda leaves x fixed (constant
# width) and displaces p by 2*lambda*G while ALSO reshaping <p^2>_lambda (parabolic, minimised at
# lambda*, exactly as verified against exact diagonalisation) -- so the ellipse both translates
# vertically AND narrows toward lambda*, widening at the Coulomb/dipole ends. ----
p_centers = [-0.85, -0.45, 0.0, 0.45, 0.85]
h_min, h_scale = 0.40, 0.85
heights = h_min + h_scale*(y_pts-ymin)
for i, (pc, ph) in enumerate(zip(p_centers, heights)):
    axe = fig.add_subplot(gs[0, i])
    axe.set_xlim(-1.5,1.5); axe.set_ylim(-1.5,1.5); axe.set_aspect("equal"); axe.axis("off")
    is_center = (i==2)
    fill_alpha = 0.85 if is_center else 0.62
    axe.add_patch(Ellipse((0,pc), width=1.0, height=ph,
                           facecolor=(*blue_rgb, fill_alpha), edgecolor=CB["blue"], linewidth=2.0))
    if is_center:
        axe.add_patch(Ellipse((0,pc), width=1.16, height=ph+0.16, facecolor="none",
                               edgecolor=CB["orange"], linewidth=1.6, linestyle=(0,(3,2)), zorder=4))
    axe.add_patch(Circle((0,pc), radius=0.055, facecolor="white", edgecolor=CB["blue"], linewidth=1.3, zorder=5))
    if i==0:
        axe.annotate("", xy=(1.35,0), xytext=(-1.35,0), clip_on=False,
                     arrowprops=dict(arrowstyle="-|>", color=CB["black"], lw=1.0))
        axe.annotate("", xy=(0,1.35), xytext=(0,-1.35), clip_on=False,
                     arrowprops=dict(arrowstyle="-|>", color=CB["black"], lw=1.0))
        axe.text(1.42, -0.10, "$x$", fontsize=9.5, color=CB["black"], va="center", clip_on=False)
        axe.text(0.10, 1.40, "$p$", fontsize=9.5, color=CB["black"], clip_on=False)

# ---- main gauge-orbit panel ----
ax = fig.add_subplot(gs[1, :])
lam = np.linspace(-1.25, 1.25, 400)
y = ymin + Q*(lam-lam_star)**2
ax.plot(lam, y, color=CB["orange"], lw=2.6, zorder=3, solid_capstyle="round")
for lp, yp in zip(lam_pts, y_pts):
    ax.plot([lp,lp],[yp, y.max()*1.14], color="#C8C8C8", lw=0.8, ls=(0,(2,3)), zorder=1)
ax.scatter([lam_star],[ymin], s=70, color=CB["orange"], edgecolor=CB["black"], linewidth=1.0, zorder=5)
ax.axhline(ymin, color=CB["gray"], lw=0.8, ls=(0,(4,3)), zorder=1)

ax.set_xlim(-1.3,1.3); ax.set_ylim(0, y.max()*1.38)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_position(("data",0))
ax.set_yticks([0]); ax.set_yticklabels(["0"], fontsize=10)
ax.tick_params(axis="y", length=3)
ax.set_xticks([-1.0,-0.2,1.0]); ax.set_xticklabels(["Coulomb", r"$\lambda_*$", "Dipole"], fontsize=10.5)
ax.tick_params(axis="x", length=3)
ax.set_xlabel(r"gauge parameter $\lambda$", fontsize=12.5, labelpad=6)

ax.text(0.01, 0.97, r"$\langle a^\dagger a\rangle_\lambda$", transform=ax.transAxes,
        fontsize=12.5, color="#111111", va="top", ha="left")
ax.text(0.01, 0.80, r"$U_\lambda=\exp[\,i\lambda G(a+a^\dagger)\,]$:  $p\to p+2\lambda G$,  $x$ unchanged",
        transform=ax.transAxes, fontsize=10.5, color="#333333", va="top", ha="left")
ax.text(0.40, 0.55, r"$\langle a^\dagger a\rangle_\lambda=\langle a^\dagger a\rangle_0+C\lambda+Q\lambda^2$",
        transform=ax.transAxes, fontsize=12, color="#111111")

ax.annotate("irreducible\nminimum", xy=(lam_star, ymin), xycoords="data",
            xytext=(lam_star+0.40, ymin+y.max()*0.30), textcoords="data",
            fontsize=10.5, color=CB["black"], ha="left", va="bottom", linespacing=1.3,
            arrowprops=dict(arrowstyle="-|>", color=CB["black"], lw=1.2,
                             connectionstyle="arc3,rad=-0.15",
                             shrinkA=0, shrinkB=4), zorder=6)

fig.suptitle("The gauge orbit of the photon dressing", fontsize=14, y=0.985)
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/00_gauge_orbit_overview.{ext}")
plt.close(fig); print("wrote 00_gauge_orbit_overview")
