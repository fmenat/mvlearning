import itertools


def create_list_options(n_views):
    #create a list with all the combination/permutation of the views. To be extracted on each index i
    return list(itertools.product([0, 1], repeat=n_views))[1:][::-1]

def possible_missing_mask(n_views):
    #all possible of combination/permutation for use of views with "n_views" number of views
    return (2**n_views)-1 #or len(create_list_options(n_views))