import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings

from xarrayvideo import xarray2video, video2xarray, plot_image, to_netcdf

def to_video(dataset_in_path, dataset_out_path, images_out_path, lossy_params, 
             lossless_params, conversion_rules, files, debug, align, ignore_existing=True):    
    #Run for all cubes
    for i, input_path in (pbar:=tqdm(enumerate(files), total=len(files))):
        try:
            #Print name
            array_id= '_'.join(input_path.stem.split('_')[1:3])
            pbar.set_description(array_id)
            
            #Check if it exists
            if (dataset_out_path / array_id).exists() and ignore_existing:
                continue

            #Load
            minicube= xr.open_dataset(input_path, engine='zarr')
            
            #Fix problems with the dataset
            if 'SCL' in minicube.variables:
                minicube['SCL']= minicube['SCL'].astype(np.uint8)
            if 'cloudmask_en' in minicube.variables: 
                minicube['cloudmask_en']= minicube['cloudmask_en'].astype(np.uint8)
                
            #We drop a variable for now, to have just 2 videos to store
            minicube= minicube.drop_vars('B07')
            
            #Align cube
            if align:
                import satalign, satalign.ecc, satalign.pcc #satalign.lgm
                import cv2
                def get_masked_mean_reference(minicube, bands, mask_name='cloudmask_en'):
                    data = minicube[bands].isel(time=slice(74, None)).to_array().transpose('variable', 'time', 'y', 'x')
                    mask = minicube[mask_name].isel(time=slice(74, None)).transpose('time', 'y', 'x')
                    data_np, mask_np= np.array(data), np.array(mask)
                    data_np[:, mask_np > 0] = np.nan
                    data_np = np.nanmean(data_np, axis=1)
                    return data_np, mask_np

                bands= ['B04','B03','B02','B8A','B06','B05']
                if 'cloudmask_en' in minicube.variables:
                    reference_image, mask= get_masked_mean_reference(minicube, bands=bands, mask_name='cloudmask_en')
                else:
                    reference_image= minicube[bands].isel(time=slice(74,None)).mean("time").to_array().transpose('variable', 'y', 'x')
                    mask= None
                datacube= minicube[bands].to_array().transpose('time', 'variable', 'y', 'x')

                syncmodel= satalign.pcc.PCC( #PCC quicker, ECC more precise
                    interpolation= cv2.INTER_CUBIC,
                    datacube=datacube, # T x C x H x W
                    reference=reference_image, # C x H x W
                    channel="mean", crop_center=110)
                new_cube, warps, errors, error_before, error_after= syncmodel.run()

                #Copy back
                minicube_aligned= minicube.copy()
                for b in bands: minicube_aligned[b]= new_cube.sel(variable=b)
                print(f'{error_before=:.6f}, {error_after=:.6f}, improvement={error_before-error_after:.6f}')
            else:
                minicube_aligned= minicube

            #Compress
            arr_dict= xarray2video(minicube_aligned, array_id, conversion_rules,
                           output_path=dataset_out_path, compute_stats=False, 
                           exceptions='ignore', loglevel='quiet',
                           )  

            #Plot image every 200
            if i%200 == 0:
                minicube_new= video2xarray(dataset_out_path, array_id) 
                plot_image(minicube_new, ['B04','B03','B02'], save_name=str(images_out_path/f'{array_id}.jpg'), 
                           show=False, mask_name='cloudmask_en' if 'cloudmask_en' in minicube.variables else None)

        except Exception as e:
            print(f'Exception processing {array_id=}: {e}')
            import traceback
            traceback.print_exc()
            if debug: raise e

        #Stop?
        if debug and i > 10: break

if __name__ == '__main__':
    #Set parameters
    dataset_in_path= Path('/scratch/users/databases/deepextremes/deepextremes-minicubes/full')
    dataset_out_path= Path('/scratch/users/databases/deepextremes-video')
    images_out_path= Path('./deepextremes_images')
    lossy_params = [
                     {'c:v': 'libx265',
                      'preset': 'medium',
                      'crf': 51,
                      'x265-params': 'qpmin=0:qpmax=0.01',
                      'tune': 'psnr',
                     },
                     {'c:v': 'libx265',
                      'preset': 'medium',
                      'crf': 7,
                      'tune': 'psnr',
                     }
                    ][1] #Choose the params
    lossless_params= { 'c:v':'ffv1' }

    conversion_rules= {
        #Sets of 3 channels are the most efficient for lossy compression
        'rgb': ( ('B04','B03','B02'), ('time','x','y'), 0, lossy_params, 12),
        'ir3': ( ('B8A','B06','B05'), ('time','x','y'), 0, lossy_params, 12),

        # Compressing 1,3, or 4 channels losslessly is efficient
        'scl': ( 'SCL', ('time','x','y'), 0, lossless_params, 8),
        'cm': ( 'cloudmask_en', ('time','x','y'), 0, lossless_params, 8),
        }
    files= list(dataset_in_path.glob('*/*.zarr'))
    debug= False
    align= False
    ignore_existing= True
    
    to_video(dataset_in_path, dataset_out_path, images_out_path, lossy_params, 
             lossless_params, conversion_rules, files, debug, align, ignore_existing=ignore_existing)