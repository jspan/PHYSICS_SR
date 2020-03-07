import numpy as np
import matplotlib.pyplot as plt

psnr = np.array([])
psnr_refine1 = np.array([])
psnr_refine2 = np.array([])

log_txt = open("../experiment/EDSR_x3/log.txt")  
#log_txt = open("../experiment/EDSR_x4_without_substitution/log.txt")
#log_txt = open("../experiment/EDSR_x4_without_substitution_without_gaussian_blur/log.txt")
line = log_txt.readline()  
while line:
    if(line[0:9]=='[Set5 x3]'):
        psnr = np.append(psnr, float(line[16:22]))
        psnr_refine1 = np.append(psnr_refine1, float(line[41:47]))
        psnr_refine2 = np.append(psnr_refine2, float(line[66:72]))
    line = log_txt.readline()
log_txt.close()

plt.xlabel('Training Epoch', fontsize=15)
plt.ylabel('Average PSNRs', fontsize=15)
plt.xlim((0, len(psnr)))
plt.ylim((32, 35))
plt.title('')
plt.grid(True, linestyle = '-.')
plt.plot(range(1,len(psnr)+1), psnr, '-r', label = 'psnr')
plt.plot(range(1,len(psnr_refine1)+1), psnr_refine1, '-b', label = 'psnr_refine1')
plt.plot(range(1,len(psnr_refine2)+1), psnr_refine2, '-g', label = 'psnr_refine2')
plt.legend(loc='lower right', fontsize=15)
plt.savefig('psnr_refine1_refine2_x3.pdf')

plt.show()
