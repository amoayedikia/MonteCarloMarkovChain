
import numpy as np
from scipy.stats import wilcoxon, ranksums

#BreakPoint = 0.1
#AUC_mcmcta = np.array([0.9,0.9,0.92,0.902,0.94,0.89,0.9,0.92,0.92])
#fs_mcmcta = np.array([0.67,0.65,0.644,0.664,0.67,0.67,0.683,0.664,0.67])

#BreakPoint = 0.45
#AUC_mcmcta = np.array([0.918,0.911,0.931,0.93,0.911,0.911,0.92,0.93,0.911])
#fs_mcmcta = np.array([0.67,0.67,0.67,0.683,0.683,0.69,0.67,0.67,0.67])

#BreakPoint = 0.8
#AUC_mcmcta = np.array([0.892,0.892,0.9,0.918,0.911,0.9,0.92,0.908,0.92])
fs_mcmcta = np.array([0.67,0.7,0.71,0.67,0.68,0.665,0.695,0.692,0.71])

AUC_roux = np.array([0.534,0.534,0.535,0.535,0.535,0.534,0.534,0.546,0.546])
AUC_gauss = np.array([0.567,0.57,0.575,0.593,0.646,0.57,0.703,0.58,0.54])
AUC_raykar = np.array([0.567,0.567,0.567,0.567,0.567,0.567,0.567,0.576,0.734])

fs_roux = np.array([0.47,0.47,0.46,0.46,0.46,0.47,0.47,0.47,0.47])
fs_gauss = np.array([0.567,0.57,0.575,0.593,0.646,0.57,0.703,0.58,0.54])
fs_raykar = np.array([0.455,0.455,0.455,0.455,0.455,0.455,0.455,0.45,0.45])

d = fs_mcmcta - fs_gauss
w, p = wilcoxon(d)
print(str(w), str(p))


