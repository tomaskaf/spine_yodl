from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
import skimage.measure
import mahotas
import numpy as np
import waterz

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        fragments_in_xy=False,
        return_seeds=False,
        min_seed_distance=5,
        mask_threshold=0.1):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True'''

    if fragments_in_xy:

        mean_affs = 0.5*(affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z]>mask_threshold*max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0)>mask_threshold*max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            return_seeds,
            min_seed_distance=min_seed_distance)

    mask = np.mean(affs, axis=0)>mask_threshold*max_affinity_value
    masked_fragments = skimage.measure.label(ret[0]*mask).astype(np.uint64)
    max_id = np.max(masked_fragments)
    ret = (masked_fragments, max_id)

    return ret


def watershed_from_boundary_distance(
        boundary_distances,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def evaluate_affs(affs, labels, thresholds):

    scores = {}
    segmentations = []

    fragments = watershed_from_affinities(affs)[0]

    a = affs.astype(np.float32)
    l = labels.astype(np.uint32)
    f = fragments.astype(np.uint64)

    i = 0
    for segmentation, metrics in waterz.agglomerate(
            affs=a,
            thresholds=thresholds,
            gt=l,
            fragments=f):
        segmentations.append(segmentation)
        scores[f'threshold_{thresholds[i]}'] = {
            'voi_split': metrics['V_Info_split'],
            'voi_merge': metrics['V_Info_merge'],
            'rand_split': metrics['V_Rand_split'],
            'rand_merge': metrics['V_Rand_merge']
        }
        i += 1

    return segmentations, scores, fragments

def segment_affs(affs, thresholds):

    #scores = {}
    segmentations = []

    fragments = watershed_from_affinities(affs)[0]

    a = affs.astype(np.float32)
    #l = labels.astype(np.uint32)
    f = fragments.astype(np.uint64)

    i = 0
    for segmentation in waterz.agglomerate(
            affs=a,
            thresholds=thresholds,
            #gt=l,
            fragments=f):
        segmentations.append(segmentation)
       # scores[f'threshold_{thresholds[i]}'] = {
          #  'voi_split': metrics['V_Info_split'],
         #  'rand_split': metrics['V_Rand_split'],
          #  'rand_merge': metrics['V_Rand_merge']
        #}
        i += 1

    return segmentations, fragments