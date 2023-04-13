import os
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder,NucleiConceptExtractor,VahadaneStainNormalizer
import numpy as np
import torch
from PIL import Image
import glob
from dgl.data.utils import save_graphs

# define normalizer
target="/byz/E2F4/target.png"
normalizer = VahadaneStainNormalizer(target_path=target)

# define graph builder functions
nuclei_detector = NucleiExtractor()
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
nuclei_concept_extractor = NucleiConceptExtractor()
knn_graph_builder = KNNGraphBuilder(k=5, thresh=75, add_loc_feats=True)

# build and save graphs
mydir="/byz/tiles/*.png"
myinf1=glob.glob(mydir)

for i in range(len(myinf1)):
    try:
        image = Image.open(myinf1[i])
        image = np.array(image.convert('RGB'))
        image = normalizer.process(image)
        nuclei_map, _ = nuclei_detector.process(image)
        features = feature_extractor.process(image, nuclei_map)
        concepts = nuclei_concept_extractor.process(image, nuclei_map)
        cell_graph = knn_graph_builder.process(nuclei_map, features)
        cell_graph.ndata['concepts'] = torch.from_numpy(concepts).to(features.device)
    except:
        print(myinf1[i].split('/')[-1],"is not processed")
    else:
        out_fname=os.path.join('/byz/E2F4/CG', myinf1[i].split('/')[-1][:-4])+".bin"
        save_graphs(
            filename=out_fname,
            g_list=[cell_graph],
        )