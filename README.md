<img src="https://i.ibb.co/54vTYzk/ssl-logo.png" alt="ssl-logo" border="0">

<p float="left"><img src="https://img.shields.io/badge/python-v3.7+-red"> <img src="https://img.shields.io/badge/pytorch-v1.7+-blue"> <img src="https://img.shields.io/badge/tensorflow-v1.14+-green">  <br>

**FuxiRec** is a Python framework for self-supervised recommendation (SSR) which integrates commonly used datasets and metrics, and implements many state-of-the-art SSR models, based on **SELFRec**. FuxiRec has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.
<br>
**Founder and principal contributor**: [@Coder-Yu ](https://github.com/Coder-Yu) [@xiaxin1998](https://github.com/xiaxin1998) <br>

This repo is released with a series of [survey papers](https://arxiv.org/abs/2203.15876) on self-supervised learning for recommender systems.

**Supported by**:<br>
Prof. Hongzhi Yin, The University of Queensland, Australia, h.yin1@uq.edu.au <br>
Prof. Shazia Sadiq, ARC Training Centre for Information Resilience (CIRES), University of Queensland, Australia

**Maintained & Updated by**:<br>
Ximing Chen, University of Macau, Macau, yc37921@um.edu.mo <br>

<h2>Architecture<h2>
<img src="https://raw.githubusercontent.com/Coder-Yu/SELFRec/main/selfrec.jpg" alt="ssl-logo" border="0" style="width:600px">


<h2>Features</h2>
<ul>
<li><b>Fast execution</b>: FuxiRec is compatible with Python 3.8+, Tensorflow 1.14+, and PyTorch 1.8+ and powered by GPUs. We also optimize the time-consuming item ranking procedure, drastically reducing ranking time to seconds. </li>
<li><b>Easy configuration</b>: FuxiRec provides simple and high-level interfaces, making it easy to add new SSR models in a plug-and-play fashion.</li>
<li><b>Highly Modularized</b>: FuxiRec is divided into multiple discrete and independent modules. This design decouples model design from other procedures, allowing users to focus on the logic of their method and streamlining development.</li>
<li><b>SSR-Specific</b>:  FuxiRec is designed specifically for SSR. It provides specific modules and interfaces for rapid development of data augmentation and self-supervised tasks.</li>
</ul>

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

<h2>Implemented Models</h2>

<table class="table table-hover table-bordered">
  <tr>
		<th>Model</th> 		<th>Paper</th>      <th>Type</th>   <th>Code</th>
   </tr>
   <tr>
    <td scope="row">SASRec</td>
        <td>Kang et al. <a href="https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf" target="_blank">Self-Attentive Sequential Recommendation</a>, ICDM'18.
         </td> <td>Sequential</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">CL4SRec</td>
        <td>Xie et al. <a href="https://arxiv.org/abs/2010.14395" target="_blank">Contrastive Learning for Sequential Recommendation</a>, ICDE'22.
         </td> <td>Sequential</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">BERT4Rec</td>
        <td>Sun et al. <a href="https://dl.acm.org/doi/pdf/10.1145/3357384.3357895" target="_blank">BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer</a>, CIKM'19.
         </td> <td>Sequential</d> <td>PyTorch</d>
      </tr>
   <table>

<table class="table table-hover table-bordered">
  <tr>
		<th>Model</th> 		<th>Paper</th>      <th>Type</th>   <th>Code</th>
   </tr>
   <tr>
    <td scope="row">ChebyCF</td>
        <td>Kim et al. <a href="https://arxiv.org/abs/2505.00552" target="_blank">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a>, SIGIR'25.
         </td> <td>Graph+DM</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">HDRM</td>
        <td>Yuan et al. <a href="https://arxiv.org/abs/2504.01541v1" target="_blank">Hyperbolic Diffusion Recommender Model</a>, WWW'25.
         </td> <td>DM</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">PreferDiff</td>
        <td>Liu et al. <a href="https://openreview.net/forum?id=6GATHdOi1x" target="_blank">Preference Diffusion for Recommendation</a>, ICLR'25.
         </td> <td>DM</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">SGCL</td>
        <td>Zhao et al. <a href="https://dl.acm.org/doi/10.1145/3722103" target="_blank">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a>, TOIS'25.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">EGCF</td>
        <td>Zhang et al. <a href="https://dl.acm.org/doi/10.1145/3701230" target="_blank">Simplify to the Limit! Embedding-Less Graph Collaborative Filtering for Recommender Systems</a>, TOIS'24.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
  <tr>
   <td scope="row">SCCF</td>
       <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/3637528.3671840" target="_blank">Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering</a>, KDD'24.
        </td> <td>Graph</d> <td>PyTorch</d>
     </tr>  
	 <tr>
    <td scope="row">SGFCF</td>
        <td>Peng et al. <a href="https://dl.acm.org/doi/10.1145/3637528.3671789" target="_blank">How Powerful is Graph Filtering for Recommendation</a>, KDD'24.
         </td> <td>Spectral Graph</d> <td>PyTorch</d>
      </tr>
    <tr>
     <td scope="row">DDRM</td>
         <td>Zhao et al. <a href="https://dl.acm.org/doi/10.1145/3626772.3657825" target="_blank">Denoising Diffusion Recommender Model</a>, SIGIR'24.
          </td> <td>Graph + DM</d> <td>PyTorch</d>
       </tr>
    <tr>
     <td scope="row">GiffCF</td>
         <td>Zhu et al. <a href="https://dl.acm.org/doi/10.1145/3626772.3657759" target="_blank">Graph Signal Diffusion Model for Collaborative Filtering</a>, SIGIR'24.
          </td> <td>Graph + DM</d> <td>PyTorch</d>
       </tr>
    <tr>
     <td scope="row">RecDCL</td>
         <td>Zhang et al. <a href="https://dl.acm.org/doi/10.1145/3589334.3645533" target="_blank">RecDCL: Dual Contrastive Learning for Recommendation</a>, WWW'24.
          </td> <td>Graph + CL</d> <td>PyTorch</d>
       </tr>   
	 <tr>
    <td scope="row">LTGNN</td>
        <td>Zhang et al. <a href="https://dl.acm.org/doi/10.1145/3589334.3645486" target="_blank">Linear-Time Graph Neural Networks for Scalable Recommendations</a>, WWW'24.
         </td> <td>Graph</d> <td>PyTorch</d>
      </tr>
    <tr>
     <td scope="row">AHNS</td>
         <td>Lai et al. <a href="https://arxiv.org/abs/2401.05191" target="_blank">Adaptive Hardness Negative Sampling for Collaborative Filtering</a>, AAAI'24.
          </td> <td>Graph + NS</d> <td>PyTorch</d>
       </tr>
   <tr>
    <td scope="row">DINS</td>
        <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/3583780.3614845" target="_blank">Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering</a>, CIKM'23.
         </td> <td>Graph + NS</d> <td>PyTorch</d>
      </tr>
   <tr>
    <td scope="row">XSimGCL</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2209.02544" target="_blank">XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation</a>, TKDE'23.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
    <tr>
     <td scope="row">BSPM</td>
         <td>Choi et al. <a href="https://dl.acm.org/doi/10.1145/3539618.3591645" target="_blank">Blurring-Sharpening Process Models for Collaborative Filtering</a>, SIGIR'23.
          </td> <td>Graph + DM</d> <td>PyTorch</d>
       </tr>
    <tr>
     <td scope="row">DiffRec</td>
         <td>Wang et al. <a href="https://dl.acm.org/doi/10.1145/3539618.3591663" target="_blank">Diffusion Recommender Model</a>, SIGIR'23.
          </td> <td>DM</d> <td>PyTorch</d>
       </tr>
		<tr>
     <td scope="row">AdaGCL</td>
         <td>Jiang et al. <a href="https://dl.acm.org/doi/10.1145/3580305.3599768" target="_blank">Adaptive Graph Contrastive Learning for Recommendation</a>, KDD'23.
          </td> <td>Graph + CL</d> <td>PyTorch</d>
       </tr>
		<tr>
     <td scope="row">JGCF</td>
         <td>Guo et al. <a href="https://dl.acm.org/doi/10.1145/3580305.3599450" target="_blank">On Manipulating Signals of User-Item Graph: A Jacobi Polynomial-based Graph Collaborative Filtering</a>, KDD'23.
          </td> <td>Spectral Graph</d> <td>PyTorch</d>
       </tr>
	 <tr>
    <td scope="row">LightGCL</td>
        <td>Cai et al. <a href="https://openreview.net/forum?id=FKXVK9dyMM" target="_blank">LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation</a>, ICLR'23.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
		<tr>
     <td scope="row">PGSP</td>
         <td>Liu et al. <a href="https://dl.acm.org/doi/10.1145/3543507.3583466" target="_blank">Personalized Graph Signal Processing for Collaborative Filtering</a>, WWW'23.
          </td> <td>Spectral Graph</d> <td>PyTorch</d>
       </tr>
	 <tr>
 	<td scope="row">DENS</td>
 			<td>Lai et al. <a href="https://dl.acm.org/doi/10.1145/3539597.3570419" target="_blank">Disentangled Negative Sampling for Collaborative Filtering</a>, WSDM'23.
 			 </td> <td>Graph + NS</d> <td>PyTorch</d>
 		</tr>
	 <tr>
   <td scope="row">SelfCF</td>
       <td>Zhou et al. <a href="https://dl.acm.org/doi/10.1145/3591469" target="_blank">SelfCF: A Simple Framework for Self-supervised Collaborative Filtering</a>, TORS'23.
        </td> <td>Graph + DA</d> <td>PyTorch</d>
     </tr>
   <tr>
    <td scope="row">SimGCL</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2112.08679" target="_blank">Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation</a>, SIGIR'22.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
		<tr>
		 <td scope="row">HCCF</td>
				 <td>Xia et al. <a href="https://dl.acm.org/doi/10.1145/3477495.3532058" target="_blank">Hypergraph Contrastive Collaborative Filtering</a>, SIGIR'22.
					</td> <td>Hypergraph + CL</d> <td>PyTorch</d>
			 </tr>
   <tr>
    <td scope="row">DirectAU</td>
        <td>Wang et al. <a href="https://arxiv.org/abs/2206.12811" target="_blank">Towards Representation Alignment and Uniformity in Collaborative Filtering</a>, KDD'22.
         </td> <td>Graph</d> <td>PyTorch</d>
      </tr>
      <tr>
       <td scope="row">CODIGEM</td>
           <td>Walker et al. <a href="https://link.springer.com/chapter/10.1007/978-3-031-10989-8_47" target="_blank">Recommendation via Collaborative Diffusion Generative Model</a>, KSEM'22.
            </td> <td>DM</d> <td>PyTorch</d>
         </tr>
   <tr>
       <td scope="row">LinkProp</td>
           <td>Fu et al. <a href="https://dl.acm.org/doi/10.1145/3487553.3524712" target="_blank">Revisiting Neighborhood-based Link Prediction for
Collaborative Filtering</a>, WWW'22.
            </td> <td>Graph</d> <td>PyTorch</d>
         </tr>      
<tr>
    <td scope="row">NCL</td>
        <td>Lin et al. <a href="https://arxiv.org/abs/2202.06200" target="_blank">Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning</a>, WWW'22.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
      </tr>
		<tr>
		    <td scope="row">LGCN</td>
		        <td>Yu et al. <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20878" target="_blank">Low-Pass Graph Convolutional Network for Recommendation</a>, AAAI'22.
		         </td> <td>Spectral Graph</d> <td>PyTorch</d>
		      </tr>
   <tr>
    <td scope="row">MixGCF</td>
        <td>Huang et al. <a href="https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf" target="_blank">MixGCF: An Improved Training Method for Graph Neural
Network-based Recommender Systems</a>, KDD'21.
         </td> <td>Graph + NS</d> <td>PyTorch</d>
      </tr>
      <tr>
     <td scope="row">HGCF</td>
         <td>Sun et al. <a href="https://dl.acm.org/doi/10.1145/3442381.3450101" target="_blank">HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering</a>, WWW'21.
          </td> <td>Graph</d> <td>PyTorch</d>
       </tr>
     <tr>
    <td scope="row">MHCN</td>
        <td>Yu et al. <a href="https://dl.acm.org/doi/abs/10.1145/3442381.3449844" target="_blank">Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation</a>, WWW'21.
         </td> <td>Social + Graph + CL</d> <td>TensorFlow</d>
      </tr>
     <tr>
    <td scope="row">SGL</td>
        <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/3404835.3462862" target="_blank">Self-supervised Graph Learning for Recommendation</a>, SIGIR'21.
         </td> <td>Graph + CL</d> <td>TensorFlow & Torch</d>
      </tr>
    <tr>
    <td scope="row">SEPT</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2106.03569" target="_blank">Socially-Aware Self-supervised Tri-Training for Recommendation</a>, KDD'21.
         </td> <td>Social + Graph + CL</d> <td>TensorFlow</d>
      </tr>
          <tr>
    <td scope="row">BUIR</td>
        <td>Lee et al. <a href="https://arxiv.org/abs/2105.06323" target="_blank">Bootstrapping User and Item Representations for One-Class Collaborative Filtering</a>, SIGIR'21.
         </td> <td>Graph + DA</d> <td>PyTorch</d>
      </tr>
        <tr>
    <td scope="row">SSL4Rec</td>
        <td>Yao et al. <a href="https://dl.acm.org/doi/abs/10.1145/3459637.3481952" target="_blank">Self-supervised Learning for Large-scale Item Recommendations</a>, CIKM'21.
	     </td> <td>Graph + CL</d>  <td>PyTorch</d>
      </tr>
    <tr>
    <td scope="row">LightGCN</td>
        <td>He et al. <a href="https://dl.acm.org/doi/10.1145/3397271.3401063" target="_blank">LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation</a>, SIGIR'20.
	     </td> <td>Graph</d>  <td>PyTorch</d>
      </tr>
		<tr>
    <td scope="row">DHCF</td>
        <td>Ji et al. <a href="https://dl.acm.org/doi/10.1145/3394486.3403253" target="_blank">Dual Channel Hypergraph Collaborative Filtering</a>, KDD'20.
	     </td> <td>Hypergraph</d>  <td>PyTorch</d>
      </tr>
      <tr>
	    <td scope="row">DiffNet++</td>
	        <td>Wu et al. <a href="https://ieeexplore.ieee.org/document/9311623" target="_blank">DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation</a>, TKDE'20.
		     </td> <td>Social + Graph</d>  <td>PyTorch</d>
	      </tr>
      <tr>
	    <td scope="row">DiffNet</td>
	        <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/3331184.3331214" target="_blank">A Neural Influence Diffusion Model for Social Recommendation</a>, SIGIR'19.
		     </td> <td>Social + Graph</d>  <td>PyTorch</d>
	      </tr>
			<tr>
	    <td scope="row">APPNP</td>
	        <td>Gasteiger et al. <a href="https://openreview.net/pdf?id=H1gL-2A9Ym" target="_blank">Predict Then Propagate: Graph Neural Networks Meet Personalized Pagerank</a>, ICLR'19.
		     </td> <td>Graph</d>  <td>PyTorch</d>
	      </tr>
        <tr>
   <td scope="row">MultiVAE</td>
       <td>Liang et al. <a href="https://dl.acm.org/doi/10.1145/3178876.3186150" target="_blank">Variational Autoencoders for Collaborative Filtering</a>, WWW'18.
      </td> <td>AE</d>  <td>PyTorch</d>
     </tr>
     <tr>
      <td scope="row">NeuMF</td>
          <td>He et al. <a href="https://dl.acm.org/doi/10.1145/3038912.3052569" target="_blank">Neural Collaborative Filtering</a>, WWW'17.
         </td> <td>-</d>  <td>PyTorch</d>
  </tr>
     <tr>
    <td scope="row">CDAE</td>
        <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/2835776.2835837" target="_blank">Collaborative Denoising Auto-Encoders for Top-N Recommender Systems</a>, WSDM'16.
       </td> <td>AE</d>  <td>PyTorch</d>
      </tr>
         <tr>
    <td scope="row">MF</td>
        <td>Yehuda et al. <a href="https://ieeexplore.ieee.org/abstract/document/5197422" target="_blank">Matrix Factorization Techniques for Recommender Systems</a>, IEEE Computer'09.
	     </td> <td>-</d>  <td>PyTorch</d>
      </tr>
  </table>  
* CL is short for contrastive learning (including data augmentation) only; DA is short for data augmentation only; NS is short for negative sampling (including CL+NS); DM is short for diffusion model; AE is short for autoencoders.

<h2>Implement Your Model</h2>

1. Create a **.conf** file for your model in the directory named conf.
2. Make your model **inherit** the proper base class.
3. **Reimplement** the following functions.
	+ *build*(), *train*(), *save*(), *predict*()
4. Register your model in **main.py**.



<h2>Related Datasets</h2>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th rowspan="2" scope="col">Data Set</th>
    <th colspan="5" scope="col" class="text-center">Basic Meta</th>
    <th colspan="3" scope="col" class="text-center">User Context</th>
    </tr>
  <tr>
    <th class="text-center">Users</th>
    <th class="text-center">Items</th>
    <th colspan="2" class="text-center">Ratings (Scale)</th>
    <th class="text-center">Density</th>
    <th class="text-center">Users</th>
    <th colspan="2" class="text-center">Links (Type)</th>
    </tr>   
   <tr>
    <td><b>Douban</b></a> </td>
    <td>2,848</td>
    <td>39,586</td>
    <td width="6%">894,887</td>
    <td width="10%">[1, 5]</td>
    <td>0.794%</td>
    <td width="4%">2,848</td>
    <td width="5%">35,770</td>
    <td>Trust</td>
    </tr>
	 <tr>
    <td><b>LastFM</b></a> </td>
    <td>1,892</td>
    <td>17,632</td>
    <td width="6%">92,834</td>
    <td width="10%">implicit</td>
    <td>0.27%</td>
    <td width="4%">1,892</td>
    <td width="5%">25,434</td>
    <td>Trust</td>
    </tr>
    <tr>
    <td><b>Yelp</b></a> </td>
    <td>19,539</td>
    <td>21,266</td>
    <td width="6%">450,884</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">19,539</td>
    <td width="5%">864,157</td>
    <td>Trust</td>
    </tr>
    <tr>
    <td><b>Amazon-Book</b></a> </td>
    <td>52,463</td>
    <td>91,599</td>
    <td width="6%">2,984,108</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>
    <tr>
    <td><b>Gowalla</b></a> </td>
    <td>29,858</td>
    <td>40,981</td>
    <td width="6%">1,027,370</td>
    <td width="10%">implicit</td>
    <td>0.08%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>
    <tr>
    <td><b>Amazon-Beauty</b></a> </td>
    <td>22,364</td>
    <td>12,102</td>
    <td width="6%">198,502</td>
    <td width="10%">implicit</td>
    <td>0.07%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>
    <tr>
    <td><b>Amazon-Kindle</b></a> </td>
    <td>138,932</td>
    <td>98,729</td>
    <td width="6%">1,909,965</td>
    <td width="10%">implicit</td>
    <td>0.01%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>
    <tr>
    <td><b>Amazon-Book</b></a> </td>
    <td>52,643</td>
    <td>91,599</td>
    <td width="6%">2,984,108</td>
    <td width="10%">implicit</td>
    <td>0.06%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>
    <tr>
    <td><b>MovieLens-10M</b></a> </td>
    <td>71,567</td>
    <td>10,681</td>
    <td width="6%">10,000,054</td>
    <td width="10%">implicit</td>
    <td>1.31%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>            
  </table>
</div>


<h2>Reference</h2>
If you find this repo helpful to your research, please cite our paper.
<p></p>

```
@article{yu2023self,
  title={Self-supervised learning for recommender systems: A survey},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and Chen, Tong and Li, Jundong and Huang, Zi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
