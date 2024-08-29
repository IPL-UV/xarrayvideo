all: 185.62Mb -> ['5.35', '3.26', '1.39']Mb (showing last 10)(5.39% of original size, 3.4466 bpppb) in 31.12s
 - params={'c:v': 'vp9', 'crf': 1, 'arnr-strength': 2, 'qmin': 0, 'qmax': 1, 'lag-in-frames': 25, 'arnr-maxframes': 7}
 - Decompression time 0.38s
 - SSIM_sat 0.999151
 - MSE_sat 0.000003
 - PSNR_sat 59.1009
 - Exp. SA 0.0041 
Metrics took 5.07s to run

------------------------------------------------

Using only 17.61% of data for metrics computation. Adjust this by modifying global variable METRICS_MAX_N=10000000.0
 - SSIM_sat 0.997700
 - MSE_sat 0.000012
 - PSNR_sat 53.8257
 - Exp. SA 0.3247 
Metrics took 3.02s to run

vs:
 - SSIM_sat 0.997890
 - MSE_sat 0.000011
 - PSNR_sat 54.3619
 - Exp. SA 0.4305 
Metrics took 5.26s to run

all2: 433.12Mb -> ['0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05', '0.05']Mb (showing last 10)(4.53% of original size, 2.8964 bpppb) in 22.21s
 - params={'codec': 'JP2OpenJPEG', 'QUALITY': '25', 'YCBCR420': 'NO', 'WRITE_METADATA': 'NO'}
 - Decompression time 10.24s
 - SSIM_sat 0.998590
 - MSE_sat 0.000010
 - PSNR_sat 54.6562
 - Exp. SA 0.0082 
Metrics took 4.80s to run

all: 433.12Mb -> ['3.62', '3.26', '1.39']Mb (showing last 10)(1.91% of original size, 1.2217 bpppb) in 25.19s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [1], 'x265-params': ['qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=10:psy-rd=0:psy-rdoq=0']}
 - Decompression time 0.77s
 - SSIM_sat 0.998664
 - MSE_sat 0.000006
 - PSNR_sat 56.6928
 - Exp. SA 0.0080 
Metrics took 4.88s to run

all: 433.12Mb -> ['7.83', '7.38', '3.67']Mb (showing last 10)(4.36% of original size, 2.7901 bpppb) in 62.57s
 - params={'c:v': 'vp9', 'crf': 0, 'arnr-strength': 2, 'qmin': 0, 'qmax': 0.5, 'lag-in-frames': 25, 'arnr-maxframes': 7}
 - Decompression time 1.30s
 - SSIM_sat 0.999799
 - MSE_sat 0.000001
 - PSNR_sat 65.6540
 - Exp. SA 0.0032 
Metrics took 30.36s to run

Cesar ---------------

all: 1060.00Mb -> ['3.57', '7.63', '7.20', '2.47']Mb (showing last 10)(1.97% of original size, 0.6301 bpppb) in 21.90s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': 'qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0'}
 - Decompression time 1.74s
 - MSE_sat 0.000007 (input saturated)
 - SNR_sat 39.4189 (input saturated)
 - PSNR_sat 57.5006 (input saturated)
 - Exp. SA 0.0108 (input saturated)
(note: using pytorch -> Same results, much much quicker!)
 - SSIM_sat 0.998445
 - MSE_sat 0.000007
 - PSNR_sat 57.5006
 - Exp. SA 0.0108 
Metrics took 22.77s to run

all: 1060.00Mb -> ['6.91', '4.08', '3.39', '2.47']Mb (showing last 10)(1.59% of original size, 0.5089 bpppb) in 16.87s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': 'qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0'}
 - Decompression time 3.53s
 - SSIM_sat 0.998070
 - MSE_sat 0.000011
 - PSNR_sat 55.7334
 - Exp. SA 0.0114 
Metrics took 22.06s to run

all: 1060.00Mb -> ['0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25']Mb (showing last 10)(4.09% of original size, 1.3088 bpppb) in 17.03s
 - params={'codec': 'JP2OpenJPEG', 'QUALITY': '5', 'YCBCR420': 'NO', 'WRITE_METADATA': 'NO'}
Reading out/cesar/all_{id}.jp2: 100%|██████████| 106/106 [00:04<00:00, 21.30it/s]
 - Decompression time 5.09s
 - SSIM_sat 0.996206
 - MSE_sat 0.000017
 - PSNR_sat 53.6105
 - Exp. SA 0.0155 
Metrics took 21.74s to run

-----------------

all: 216.56Mb -> ['4.62', '1.57']Mb (2.86% of original size, 0.9148 bpppb) in 20.56s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0', 'qpmax=10:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.06s
 - MSE_sat 0.000012 (input saturated)
 - SNR_sat 42.0605 (input saturated)
 - PSNR_sat 59.5316 (input saturated)
 - Exp. SA 0.0092 (input saturated)
 
 all: 216.56Mb -> ['4.62', '3.18']Mb (3.60% of original size, 1.1536 bpppb) in 24.13s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=10:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.16s
 - MSE_sat 0.000011 (input saturated)
 - SNR_sat 42.4629 (input saturated)
 - PSNR_sat 59.9340 (input saturated)
 - Exp. SA 0.0083 (input saturated)

all2: 216.56Mb -> ['3.27', '3.00', '1.94']Mb (3.79% of original size, 1.2141 bpppb) in 24.32s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0']}
 - Decompression time 0.73s
 - MSE_sat 0.000012 (input saturated)
 - SNR_sat 41.9893 (input saturated)
 - PSNR_sat 54.0777 (input saturated)
 - Exp. SA 0.0096 (input saturated)

all: 216.56Mb -> ['4.62', '1.17', '0.61']Mb (2.95% of original size, 0.9452 bpppb) in 23.01s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=1:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=12:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.25s
 - MSE_sat 0.000010 (input saturated)
 - SNR_sat 42.7201 (input saturated)
 - PSNR_sat 60.1912 (input saturated)
 - Exp. SA 0.0088 (input saturated)

all: 216.56Mb -> ['4.55', '1.17', '0.61']Mb (2.92% of original size, 0.9355 bpppb) in 23.05s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=2:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=12:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.26s
 - MSE_sat 0.000010 (input saturated)
 - SNR_sat 42.6150 (input saturated)
 - PSNR_sat 60.0861 (input saturated)
 - Exp. SA 0.0088 (input saturated)

all: 216.56Mb -> ['3.56', '2.58', '1.26']Mb (3.42% of original size, 1.0930 bpppb) in 27.34s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.25s
 - MSE_sat 0.000014 (input saturated)
 - SNR_sat 41.2683 (input saturated)
 - PSNR_sat 58.7394 (input saturated)
 - Exp. SA 0.0083 (input saturated)

all: 216.56Mb -> ['4.14', '2.02', '1.02']Mb (3.32% of original size, 1.0608 bpppb) in 26.36s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=4:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=6:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=8:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.28s
 - MSE_sat 0.000011 (input saturated)
 - SNR_sat 42.3502 (input saturated)
 - PSNR_sat 59.8213 (input saturated)
 - Exp. SA 0.0081 (input saturated)

all: 216.56Mb -> ['3.86', '1.93', '0.99']Mb (3.13% of original size, 1.0003 bpppb) in 25.31s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': ['qpmin=0:qpmax=5:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=10:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=20:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.14s
 - MSE_sat 0.000013 (input saturated)
 - SNR_sat 41.7052 (input saturated)
 - PSNR_sat 59.1763 (input saturated)
 - Exp. SA 0.0085 (input saturated)
 
all: 216.56Mb -> ['4.15', '1.17', '0.61']Mb (2.74% of original size, 0.8756 bpppb) in 22.42s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=4:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=8:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=12:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.21s
 - MSE_sat 0.000012 (input saturated)
 - SNR_sat 41.8994 (input saturated)
 - PSNR_sat 59.3705 (input saturated)
 - Exp. SA 0.0091 (input saturated)
 
 all: 216.56Mb -> ['4.63', '0.65', '0.35']Mb (2.59% of original size, 0.8301 bpppb) in 20.15s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=0.2:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=4:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=16:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.08s
 - MSE_sat 0.000012 (input saturated)
 - SNR_sat 41.9540 (input saturated)
 - PSNR_sat 59.4251 (input saturated)
 - Exp. SA 0.0099 (input saturated)

all: 216.56Mb -> ['3.86', '0.30', '0.18']Mb (2.00% of original size, 0.6410 bpppb) in 16.71s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [50], 'x265-params': ['qpmin=0:qpmax=5:psy-rd=0:psy-rdoq=0', 'qpmin=0:qpmax=10:psy-rd=0:psy-rdoq=0qpmin=0:qpmax=20:psy-rd=0:psy-rdoq=0']}
 - Decompression time 1.15s
 - MSE_sat 0.000018 (input saturated)
 - SNR_sat 40.0357 (input saturated)
 - PSNR_sat 57.5068 (input saturated)
 - Exp. SA 0.0115 (input saturated)
----------------






rgb: 92.81Mb -> ['2.36']Mb (2.54% of original size, 0.8132 bpppb) in 7.35s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=1'}
 - Decompression time 0.24s
 - MSE_sat 0.000010 (input saturated)
 - SNR_sat 42.8880 (input saturated)
 - PSNR_sat 54.9945 (input saturated)
 - Exp. SA 0.0141 (input saturated)

ir3: 92.81Mb -> ['2.62']Mb (2.82% of original size, 0.9027 bpppb) in 8.29s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=1'}
 - Decompression time 0.30s
 - MSE_sat 0.000018 (input saturated)
 - SNR_sat 40.2383 (input saturated)
 - PSNR_sat 51.5264 (input saturated)
 - Exp. SA 0.0097 (input saturated)
 
 
 
 
 
 

 ---------------------------------------------------------------------------------------------
 
 
 
 
Stacking (crf 4)

all2: 216.56Mb -> ['1.48', '1.47', '1.37']Mb (1.99% of original size, 0.6373 bpppb) in 4.40s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 0.63s
 - MSE_sat 0.000026 (input saturated)
 - SNR_sat 38.4939 (input saturated)
 - PSNR_sat 50.5767 (input saturated)
 - Exp. SA 0.0177 (input saturated)
 
PCA 6 PCs (crf 4)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%']
all: 216.56Mb -> ['2.10', '1.68']Mb (1.74% of original size, 0.5584 bpppb) in 5.00s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 0.44s
 - MSE_sat 0.000038 (input saturated)
 - SNR_sat 36.9448 (input saturated)
 - PSNR_sat 54.4102 (input saturated)
 - Exp. SA 0.0153 (input saturated)

PCA 6 PCs (crf 00)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%']
all: 216.56Mb -> ['3.11', '2.95']Mb (2.80% of original size, 0.8955 bpppb) in 6.04s
 - MSE_sat 0.000019 (input saturated)
 - SNR_sat 39.9475 (input saturated)
 - PSNR_sat 57.4128 (input saturated)
 - Exp. SA 0.0119 (input saturated)

PCA 7 PCs (crf 4)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['2.10', '1.68', '1.40']Mb (2.39% of original size, 0.7653 bpppb) in 6.67s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.08s
 - MSE_sat 0.000035 (input saturated)
 - SNR_sat 37.3263 (input saturated)
 - PSNR_sat 54.7917 (input saturated)
 - Exp. SA 0.0143 (input saturated)

---------------------------------------------------------------------------------------------
PCA 7 PCs (crf 0 0 12)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['3.12', '2.94', '0.43']Mb (2.99% of original size, 0.9581 bpppb) in 7.27s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 12, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.15s
 - MSE_sat 0.000016 (input saturated)
 - SNR_sat 40.6069 (input saturated)
 - PSNR_sat 58.0722 (input saturated)
 - Exp. SA 0.0112 (input saturated)

---------------------------------------------------------------------------------------------
PCA 7 PCs (crf 0 1 2)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['3.12', '2.58', '1.66']Mb (3.40% of original size, 1.0869 bpppb) in 7.86s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 2, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.17s
 - MSE_sat 0.000016 (input saturated)
 - SNR_sat 40.6897 (input saturated)
 - PSNR_sat 58.1551 (input saturated)
 - Exp. SA 0.0108 (input saturated)

---------------------------------------------------------------------------------------------
PCA 7 PCs (crf 0 1 12)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['3.12', '2.60', '0.43']Mb (2.84% of original size, 0.9080 bpppb) in 6.89s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 12, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.22s
 - MSE_sat 0.000016 (input saturated)
 - SNR_sat 40.5401 (input saturated)
 - PSNR_sat 58.0055 (input saturated)
 - Exp. SA 0.0114 (input saturated)

PCA 7 PCs (crf 028)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['3.12', '2.26', '0.88']Mb (2.89% of original size, 0.9237 bpppb) in 7.22s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 8, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.19s
 - MSE_sat 0.000016 (input saturated)
 - SNR_sat 40.5543 (input saturated)
 - PSNR_sat 58.0197 (input saturated)
 - Exp. SA 0.0113 (input saturated

PCA 7 PCs (crf 036)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['3.12', '1.95', '1.14']Mb (2.87% of original size, 0.9172 bpppb) in 7.37s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 6, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.00s
 - MSE_sat 0.000017 (input saturated)
 - SNR_sat 40.4851 (input saturated)
 - PSNR_sat 57.9504 (input saturated)
 - Exp. SA 0.0114 (input saturated)

PCA 7 PCs (crf 135)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['2.83', '1.95', '1.27']Mb (2.80% of original size, 0.8945 bpppb) in 7.13s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 5, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.05s
 - MSE_sat 0.000020 (input saturated)
 - SNR_sat 39.7470 (input saturated)
 - PSNR_sat 57.2123 (input saturated)
 - Exp. SA 0.0120 (input saturated)

PCA 7 PCs (crf 234)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['2.57', '1.96', '1.40']Mb (2.74% of original size, 0.8766 bpppb) in 7.03s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.05s
 - MSE_sat 0.000024 (input saturated)
 - SNR_sat 38.9740 (input saturated)
 - PSNR_sat 56.4394 (input saturated)
 - Exp. SA 0.0126 (input saturated)
 
PCA 7 PCs (crf 333)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['2.33', '1.96', '1.53']Mb (2.68% of original size, 0.8591 bpppb) in 7.10s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 3, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 0.98s
 - MSE_sat 0.000028 (input saturated)
 - SNR_sat 38.1879 (input saturated)
 - PSNR_sat 55.6533 (input saturated)
 - Exp. SA 0.0133 (input saturated)

 PCA 7 PCs (crf 432)

Explained variance: ['99.4372%', '0.4190%', '0.1018%', '0.0173%', '0.0151%', '0.0052%', '0.0044%']
all: 216.56Mb -> ['2.10', '1.95', '1.66']Mb (2.64% of original size, 0.8432 bpppb) in 6.90s
 - params={'c:v': 'libx264', 'preset': 'slow', 'crf': 2, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 1.05s
 - MSE_sat 0.000034 (input saturated)
 - SNR_sat 37.3808 (input saturated)
 - PSNR_sat 54.8462 (input saturated)
 - Exp. SA 0.0141 (input saturated)





 ---------------------------------------------------------------------------------------------
 
 
 
 
wind_speed: 14622.43Mb -> ['35.30', '44.08', '57.67', '86.65', '67.06']Mb (1.99% of original size, 0.6363 bpppb) in 191.85s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=0.01'}
 - Decompression time 12.13s
 - MSE_sat 0.019001 (input saturated)
 - SNR_sat 42.8042 (input saturated)
 - PSNR_sat 58.7520 (input saturated)
 - Exp. SA 0.0122 (input saturated)
 
wind_speed: 14622.43Mb -> ['45.92', '56.85', '74.96', '114.02', '65.12']Mb (2.44% of original size, 0.7810 bpppb) in 209.95s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=0.001:psy-rd=0:psy-rdoq=0'}
 - Decompression time 12.67s
 - MSE_sat 0.011255 (input saturated)
 - SNR_sat 45.0786 (input saturated)
 - PSNR_sat 61.0263 (input saturated)
 - Exp. SA 0.0097 (input saturated)

wind_u: 14622.43Mb -> ['27.25', '32.30', '38.76', '53.21', '42.01']Mb (1.32% of original size, 0.4235 bpppb) in 166.98s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=0.01'}
 - Decompression time 9.94s
 - MSE_sat 0.037294 (input saturated)
 - SNR_sat 38.5796 (input saturated)
 - PSNR_sat 55.3303 (input saturated)
 - Exp. SA 0.0233 (input saturated)
 
 
 
 
 
 
 
 ---------------------------------------------------------------------------------------------
 
 
 
 
 
 
 Repeating last band (13->15 bands)

wind_speed: 9500.56Mb -> ['123.91', '150.33', '148.91', '163.44', '94.47']Mb (7.17% of original size, 2.2940 bpppb) in 309.20s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 51.12s
 - MSE_sat 0.396772 (input saturated)
 - SNR_sat 29.1306 (input saturated)
 - PSNR_sat 45.0145 (input saturated)
 - Exp. SA 0.0459 (input saturated)

PCA (13->12 bands)

wind_speed2: 9500.56Mb -> ['156.51', '184.56', '200.42', '203.73']Mb (7.84% of original size, 2.5101 bpppb) in 323.20s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 4, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 40.36s
 - MSE_sat 0.395984 (input saturated)
 - SNR_sat 29.1392 (input saturated)
 - PSNR_sat 49.8067 (input saturated)
 - Exp. SA 0.0414 (input saturated)


PCA (13->15 bands) (crf [0,1,3,6,9])
wind_speed: 9500.56Mb -> ['207.37', '224.37', '213.70', '177.33', '77.13']Mb (9.47% of original size, 3.0311 bpppb) in 383.62s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 9, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 70.17s
 - MSE_sat 0.158937 (input saturated)
 - SNR_sat 33.1038 (input saturated)
 - PSNR_sat 53.7713 (input saturated)
 - Exp. SA 0.0270 (input saturated)
PCA (13->15 bands) (crf [1,3,6,9,12])

Explained variance: ['71.6457%', '12.6993%', '8.5486%', '3.2135%', '1.6074%', '0.8591%', '0.5308%', '0.3500%', '0.2053%', '0.1390%', '0.0958%', '0.0589%', '0.0466%']
wind_speed3: 9500.56Mb -> ['194.14', '197.66', '173.68', '136.79', '63.81']Mb (8.06% of original size, 2.5803 bpppb) in 366.19s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 12, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 66.76s
 - MSE_sat 0.236834 (input saturated)
 - SNR_sat 31.3716 (input saturated)
 - PSNR_sat 52.0391 (input saturated)
 - Exp. SA 0.0339 (input saturated)
---
PCA (13->15 bands) (crf [0,1,3,6,9])
wind_u: 9500.56Mb -> ['83.75', '106.09', '124.92', '137.57', '72.48']Mb (5.52% of original size, 1.7677 bpppb) in 315.69s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 9, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 58.64s
 - MSE_sat 1.637260 (input saturated)
 - SNR_sat 21.6157 (input saturated)
 - PSNR_sat 44.3922 (input saturated)
 - Exp. SA 0.1190 (input saturated)
 ---
 Higher res
 wind_speed: 39954.24Mb -> ['203.22', '277.62', '303.11', '267.91', '140.02']Mb (2.98% of original size, 0.9546 bpppb) in 144.77s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 9, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 150.13s
 - MSE_sat 0.118003 (input saturated)
 - SNR_sat 34.7733 (input saturated)
 - PSNR_sat 54.8548 (input saturated)
 - Exp. SA 0.0273 (input saturated)
 
 wind_u: 39954.24Mb -> ['68.26', '122.58', '153.96', '183.57', '131.28']Mb (1.65% of original size, 0.5283 bpppb) in 127.51s
 - params={'c:v': 'libx264', 'preset': 'medium', 'crf': 9, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - MSE_sat 0.471016 (input saturated)
 - SNR_sat 27.5321 (input saturated)
 - PSNR_sat 49.7138 (input saturated)
 - Exp. SA 0.0640 (input saturated)
 
 ---
 Higher res, but using just a small subset of time samples
 
 wind_u: 14622.43Mb -> ['26.00', '30.33', '36.15', '48.83', '38.17']Mb (1.23% of original size, 0.3928 bpppb) in 175.21s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=4'}
 - Decompression time 10.28s
 - MSE_sat 0.039448 (input saturated)
 - SNR_sat 38.3357 (input saturated)
 - PSNR_sat 55.0864 (input saturated)
 - Exp. SA 0.0237 (input saturated)
 
 wind_u: 14622.43Mb -> ['27.25', '32.29', '38.74', '53.17', '41.92']Mb (1.32% of original size, 0.4232 bpppb) in 167.06s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=2'}
 - Decompression time 10.13s
 - MSE_sat 0.037310 (input saturated)
 - SNR_sat 38.5777 (input saturated)
 - PSNR_sat 55.3284 (input saturated)
 - Exp. SA 0.0233 (input saturated)
 
 
 wind_u: 14622.43Mb -> ['27.25', '32.30', '38.76', '53.21', '42.01']Mb (1.32% of original size, 0.4235 bpppb) in 166.98s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=0.01'}
 - Decompression time 9.94s
 - MSE_sat 0.037294 (input saturated)
 - SNR_sat 38.5796 (input saturated)
 - PSNR_sat 55.3303 (input saturated)
 - Exp. SA 0.0233 (input saturated)
 
 PCA Explained variance: ['71.7423%', '9.7276%', '8.7947%', '3.9081%', '2.0796%', '1.1597%', '0.8252%', '0.5898%', '0.3831%', '0.2860%', '0.2379%', '0.1482%', '0.1178%']
 
 wind_speed: 14622.43Mb -> ['35.28', '62.50', '77.76', '98.15', '66.26']Mb (2.32% of original size, 0.7439 bpppb) in 206.57s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=2'}
 - Decompression time 45.06s
 - MSE_sat 0.023781 (input saturated)
 - SNR_sat 41.8297 (input saturated)
 - PSNR_sat 61.8949 (input saturated)
 - Exp. SA 0.0141 (input saturated)
 
 wind_speed: 14622.43Mb -> ['35.30', '44.08', '57.67', '86.65', '67.06']Mb (1.99% of original size, 0.6363 bpppb) in 191.85s
 - params={'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 'x265-params': 'qpmin=0:qpmax=0.01'}
 - Decompression time 12.13s
 - MSE_sat 0.019001 (input saturated)
 - SNR_sat 42.8042 (input saturated)
 - PSNR_sat 58.7520 (input saturated)
 - Exp. SA 0.0122 (input saturated)