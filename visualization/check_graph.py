import os
import sys
sys.path.remove('/opt/apps/intel18/impi18_0/python2/2.7.16/lib/python2.7/site-packages')
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder,NucleiConceptExtractor,VahadaneStainNormalizer
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization
target="/work/07034/byz/maverick2/GNN/target.png"
normalizer = VahadaneStainNormalizer(target_path=target)
visualizer = OverlayGraphVisualization(instance_visualizer=InstanceImageVisualization(instance_style="filled+outline"))
nuclei_detector = NucleiExtractor()
feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
nuclei_concept_extractor = NucleiConceptExtractor()
knn_graph_builder = KNNGraphBuilder(k=5, thresh=75, add_loc_feats=True)
from dgl.data.utils import save_graphs
myinf="/work/07034/byz/maverick2/Transfer/*.png"
myinf=glob.glob(myinf)
for i in tqdm(range(len(myinf))):
    try:
        image= Image.open(myinf[i])
        image = np.array(image.convert('RGB'))
        image = normalizer.process(image)
        nuclei_map, _ = nuclei_detector.process(image)
        features = feature_extractor.process(image, nuclei_map)
        #cell_graph = knn_graph_builder.process(nuclei_map, features)
        #concepts = nuclei_concept_extractor.process(image, nuclei_map)
        cell_graph = knn_graph_builder.process(nuclei_map, features)
    except:
        print(myinf[i], "not processed")
    else:
        #out_fname = os.path.join('/work/07034/byz/maverick2/GNN/eg_cg', myinf[i][30:-4]) + ".bin"
        #save_graphs(
            #filename=out_fname,
            #g_list=[cell_graph],
        #)
        out_fname = os.path.join('/work/07034/byz/maverick2/exp/', myinf[i].rsplit("/",1)[1])
        canvas = visualizer.process(
            canvas=image,
            graph=cell_graph,
            instance_map=nuclei_map)
        canvas.save(out_fname)



