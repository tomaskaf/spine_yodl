from model import SpineUNet
from segment_dev import evaluate_affs
from train import AddChannelDim, RemoveChannelDim
import gunpowder as gp
import gunpowder.torch as gp_torch
import logging
import time
import zarr
#%%
logging.basicConfig(level=logging.INFO)

def predict(iteration,path_to_dataGP):
   
  
    input_size = (8, 96, 96)
    output_size = (4, 64, 64)
    amount_size = gp.Coordinate((2, 16, 16))
    model = SpineUNet(crop_output='output_size')

    raw = gp.ArrayKey('RAW')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

                                
    reference_request = gp.BatchRequest()
    reference_request.add(raw, input_size)
    reference_request.add(affs_predicted, output_size)
    
    source = gp.ZarrSource(
        path_to_dataGP,
        {
            raw: 'validate/sample1/raw'
        } 
    )
  
    with gp.build(source):
        source_roi = source.spec[raw].roi
    request = gp.BatchRequest()
    request[raw] = gp.ArraySpec(roi=source_roi)
    request[affs_predicted] = gp.ArraySpec(roi=source_roi)

    pipeline = (
        source +
       
        gp.Pad(raw,amount_size) +
        gp.Normalize(raw) +
        # raw: (d, h, w)
        gp.Stack(1) +
        # raw: (1, d, h, w)
        AddChannelDim(raw) +
        # raw: (1, 1, d, h, w)
        gp_torch.Predict(
            model,
            inputs={'x': raw},
            outputs={0: affs_predicted},
            checkpoint=f'C:/Users/filip/spine_yodl/model_checkpoint_{iteration}') +
        RemoveChannelDim(raw) +
        RemoveChannelDim(raw) +
        RemoveChannelDim(affs_predicted) +
        # raw: (d, h, w)
        # affs_predicted: (3, d, h, w)
        gp.Scan(reference_request)
    )

    with gp.build(pipeline):
        prediction = pipeline.request_batch(request)

    return prediction[raw].data, prediction[affs_predicted].data
#%%

if __name__ == "__main__":
    #start_time = time.time()
    iteration = 100000
    thresholds = [10, 20, 30]

    raw, affs_predicted = predict(iteration)

    f = zarr.open('validate.zarr')
    f[f'{iteration}/raw'] = raw
    f[f'{iteration}/affs_predicted'] = affs_predicted

    labels = zarr.open('data/20200201.zarr')['train/sample3/labels'][:]

    segmentations, scores, fragments = evaluate_affs(
        affs_predicted,
        labels,
        thresholds=thresholds)
    
    for segmentation, threshold in zip(segmentations, thresholds):
        f[f'{iteration}/segmentation_{threshold}'] = segmentation
    f[f'{iteration}/fragments'] = fragments
#print("--- %s seconds ---" % (time.time() - start_time))