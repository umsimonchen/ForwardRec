<div align="center">
  <h1>
    <strong>"ForwardRec": Aggregate, Optimize, and Forward: A Forward-Forward Algorithm Framework for
Graph-Based Recommendation</strong>
  </h1>
<img width="3289" height="1064" alt="fig_main" src="https://github.com/user-attachments/assets/d73cc8a7-65b2-4d53-a846-be65b443fbe9" />
</div>

**ForwardRec**, a novel graph recommendation framework based on the Forward-Forward (FF) algorithm built on the [SELFRec](https://github.com/Coder-Yu/SELFRec), introduces two core components. **Forward Learning**: ForwardRec tackles semantic and frequency issues by disentangling each layer independently; **Hierarchical Rejection Sampling (HRS)**: To provide precise and comprehensive semantics for optimization, HRS conducts a layer-specific hierarchical negative sampling.

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
We provide the variants shown in ablation analysis for LightGCN, LGCN, DHCF, SGL, and MixGCF. The codes are placed in the model file along with their original codes. Please uncomment the corresponding code for proper running.

<h2>Supplementary Experiments</h2>
Regarding all case studies and supplementary experiments, please refer to the folder "ForwardRec_exp/".

<h2>Large Datasets</h2>
Due to the large files, please download the MovieLens-10M on https://files.grouplens.org/datasets/movielens/ml-10m.zip and place it into the folder "dataset/ml-10M" and split the dataset into the training set and the test set by "split.py".

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
    <td scope="row">SGCL</td>
        <td>Zhao et al. <a href="https://dl.acm.org/doi/10.1145/3722103" target="_blank">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a>, TOIS'25.
         </td> <td>Graph + CL</d> <td>PyTorch</d>
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
     <td scope="row">AHNS</td>
         <td>Lai et al. <a href="https://arxiv.org/abs/2401.05191" target="_blank">Adaptive Hardness Negative Sampling for Collaborative Filtering</a>, AAAI'24.
          </td> <td>Graph + NS</d> <td>PyTorch</d>
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
    <td scope="row">MixGCF</td>
        <td>Huang et al. <a href="https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf" target="_blank">MixGCF: An Improved Training Method for Graph Neural
Network-based Recommender Systems</a>, KDD'21.
         </td> <td>Graph + NS</d> <td>PyTorch</d>
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
* CL is short for contrastive learning (including data augmentation) only; DA is short for data augmentation only; NS is short for negative sampling (including CL+NS); DM is short for diffusion model (diffusion-like, e.g., flow models); AE is short for autoencoders.

<h2>Reference</h2>
**FuxiRec** is a Python framework based on **SELFRec**, which integrates commonly used datasets and metrics, and implements many state-of-the-art models. It has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.
<br>
If you find this repo helpful to your research, please cite our paper and the base framework from Yu.
<p></p>

```
@article{chen2026tri,
  title={Towards a Tri-View Diffusion Framework for Recommendation},
  author={Chen, Ximing and Lei, Pui Ieng and Sheng, Yijun and Liu, Yanyan and Gong, Zhiguo},
  journal={The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026},
  publisher={ACM}
}
@article{yu2023self,
  title={Self-supervised learning for recommender systems: A survey},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and Chen, Tong and Li, Jundong and Huang, Zi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
