#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import numpy as np
import matplotlib.pyplot as plt

def plot_pair(img1, img2, max_val, factor=5, transpose=None, 
              titles=['Original', 'Compressed'], figsize=(8,4)):
    f, axes= plt.subplots(1,2,figsize=figsize)
    if transpose is not None:
        for i, img in enumerate([img1, img2]):
            axes[i].imshow(np.minimum(np.transpose(img, transpose)*factor, max_val)/max_val) 
    else:
        for i, img in enumerate([img1, img2]):
            axes[i].imshow(np.minimum(img*factor, max_val)/max_val)
    plt.grid(False)
    # plt.axis('off')
    for i in range(2): axes[i].set_title(titles[i])
    plt.show()
    
def plot_image(x, band_names, mask_name='cloudmask_en', t_name='time', 
               txy_coords=('time', 'variable', 'y', 'x'),
               save_name='default.jpg', cmaps=[None], show=True, plot_idx= [[0,1,2]], 
               limits=[(0,0.3)], ylabels=['RGB'], stack_every=73):
    
    from IPython.display import display, Image, HTML
    from txyvis import plot_maps
    import cv2
    
    txy= x[band_names].to_array().transpose(*txy_coords).values #t,y,x,c -> t,c,y,x
    
    if mask_name is None:
        masks= None
    else:
        #t,y,x -> t,1,y,x. Repat mask len(images) times
        masks= [x[[mask_name]].to_array().transpose(*txy_coords).values > 0]*len(plot_idx)
        
    try:
        xlabels= list(map(lambda i_d: f'{i_d[0]} {np.datetime_as_string((i_d[1]), unit="D")}',
                          enumerate(x[t_name].values)))
    except:
        xlabels= np.arange(len(x.shape[1]))
                
    img_comp= plot_maps(
            images=[txy[:,idx] for idx in plot_idx], 
            masks=masks,
            cmaps=cmaps, 
            limits= [(0,0.3)],
            ylabels=ylabels, 
            xlabels=xlabels,
            mask_kwargs=dict(colors= {0:None, 1:'r'}), classes= {1:'Invalid'},
            title=save_name, 
            backend='numpy',
            numpy_backend_kwargs={'size':13, 'color':'black', 'xstep':4,
                                  'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
            plot_mask_channel=None, matplotlib_backend_kwargs={'text_size':20},
            figsize=(27.5,10),
            stack_every=stack_every, #Stack every year (approx.) 73*5=365
                  )
    
    Path(save_name).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_name, img_comp[...,[2,1,0]])
    if show:
        display(HTML(f'<h4>{Path(save_name).stem}</h4>'))
        display(Image(save_name))