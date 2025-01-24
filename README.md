<h1>ForwardRec</h1>
<h2>Requirements</h2>
	
```
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
tensorflow==1.14.0
torch>=1.7.0
```

<h2>Usage</h2>
<ol>
<li>Configure the xx.conf file in the directory named conf. (xx is the name of the model you want to run)</li>
<li>Run main.py and choose the model you want to run.</li>
</ol>

<h2>Ablated Variants</h2>
We provide the variants shown in ablation analysis for LightGCN, LGCN, DGCF, SGL and MixGCF. The codes are placed in their model file along with their original codes. Please anti-comment the corrsponding codes when using.

<h2>Large Datasets</h2>
Please download the MovieLens-10M on https://files.grouplens.org/datasets/movielens/ml-10m.zip and use place them into folder "dataset/ml-10M" and split the dataset by the training set and the test set by "split.py".
