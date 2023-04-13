import sys
sys.path.remove('/opt/apps/intel18/impi18_0/python2/2.7.16/lib/python2.7/site-packages')
import os
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder,NucleiConceptExtractor,VahadaneStainNormalizer
import numpy as np
import shutil
from PIL import Image
import glob
import torch
from dgl.data.utils import save_graphs
target="/work/07034/byz/maverick2/GNN/target.png"
normalizer = VahadaneStainNormalizer(target_path=target)
nuclei_detector = NucleiExtractor()
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
nuclei_concept_extractor = NucleiConceptExtractor()
knn_graph_builder = KNNGraphBuilder(k=5, thresh=75, add_loc_feats=True)
mydir1="/work/07034/byz/maverick2/GNN/LUAD_DX_part2a/*.png"
#mydir2="/work/07034/byz/maverick2/GNN/BC_IDC/*.JPG"
myinf1=glob.glob(mydir1)
#myinf2=glob.glob(mydir2)
#myinf1.extend(myinf2)
processed="/work/07034/byz/maverick2/GNN/LUAD_DX_part2_finished"
unprocessed="/work/07034/byz/maverick2/GNN/LUAD_DX_part2_failed"
for i in range(len(myinf1)):
    try:
        image = Image.open(myinf1[i])
        image = np.array(image.convert('RGB'))
        image = normalizer.process(image)
        nuclei_map, _ = nuclei_detector.process(image)
        #features = feature_extractor.process(image, nuclei_map)
        #cell_graph = knn_graph_builder.process(nuclei_map, features)
        concepts = nuclei_concept_extractor.process(image, nuclei_map)
        cell_graph = knn_graph_builder.process(nuclei_map, concepts)
        #cell_graph.ndata['concepts'] = torch.from_numpy(concepts).to(features.device)
    except:
        print(myinf1[i],"not processed")
        outfile = os.path.join(unprocessed, myinf1[i][45:])
        shutil.move(myinf1[i], outfile)
    else:
        out_fname=os.path.join('/work/07034/byz/maverick2/GNN/LUAD_CG', myinf1[i][45:-4])+".bin"
        save_graphs(
            filename=out_fname,
            g_list=[cell_graph],
        )
        outfile = os.path.join(processed, myinf1[i][45:])
        shutil.move(myinf1[i], outfile)