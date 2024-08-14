import numpy as np

from .utils import possible_missing_mask, create_list_options

def augment_randomlist_missing(view_names_forward):
    #return a single augmentation combination based on missing
    n_views = len(view_names_forward)
    n_missing_mask = possible_missing_mask(n_views)
    lst_missing = create_list_options(n_views)
    
    i_rnd = np.random.randint(0, n_missing_mask)
    views_mask_i = np.asarray(lst_missing[i_rnd]).astype(bool) #mask as [0,1,1]
    affected_views =  np.asarray(view_names_forward)[views_mask_i].tolist()

    return affected_views

def augment_random_missing(view_names_forward, drop_ratio= 0.5):
    #return a single augmentation combination based on missing
    assert drop_ratio > 0 and drop_ratio < 1, "Drop ratio should be between 0 and 1"

    affected_views = []
    while (len(affected_views) == 0):
        for v in view_names_forward:
            if np.random.rand() > drop_ratio:
                affected_views.append(v)        
    return affected_views



if __name__ == "__main__":
    view_names = [ "S1", "S2","weather"]
    
    print("View names =",view_names)
    print("Random augmented views from list = ",augment_randomlist_missing(view_names))

    print("Random with ratio 10 of augmented views = ",augment_random_missing(view_names, drop_ratio=0.10))
    print("Random with ratio 30 of augmented views = ",augment_random_missing(view_names, drop_ratio=0.30))
    print("Random with ratio 50 of augmented views = ",augment_random_missing(view_names, drop_ratio=0.50))
    print("Random with ratio 70 of augmented views = ",augment_random_missing(view_names, drop_ratio=0.70))
    print("Random with ratio 90 of augmented views = ",augment_random_missing(view_names, drop_ratio=0.90))
