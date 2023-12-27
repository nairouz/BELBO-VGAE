# Beyond The Evidence Lower Bound: Dual Variational Graph Auto-Encoders For Node Clustering (BELBO-VGAE)

## Abstract

Variational Graph Auto-Encoders (VGAEs) have achieved promising performance in several applications. Some recent models incorporate the clustering inductive bias by imposing non-Gaussian prior distributions. However, the regularization term is practically insufficient to learn the clustering structures due to the mismatch between the target and the learned distributions. Thus, we formulate a new variational lower bound that incorporates an explicit clustering objective function. The introduction of a clustering objective leads to two problems. First, the latent information destroyed by the clustering process is critical for generating the between-cluster edges. Second, the noisy and sparse input graph does not benefit from the information learned during the clustering process. To address the first problem, we identify a new term overlooked by existing Evidence Lower BOunds (ELBOs). This term accounts for the difference between the variational posterior used for the clustering task and the variational posterior associated with the generation task. Furthermore, we find that the new term increases resistance to posterior collapse. Theoretically, we demonstrate that our lower bound is a tighter approximation of the log-likelihood function. To address the second problem, we propose a graph update algorithm that reduces the over-segmentation and under-segmentation problems. We conduct several experiments to validate the merits of our approach. Our results show that the proposed method considerably improves the clustering quality compared to state-of-the-art VGAE models.

## Conceptual design

<p align="center">
<img align="center" src="https://github.com/nairouz/BELBO-VGAE/tree/main/images/model_BELBO.png">
</p>

## Some Results

### Quantitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/BELBO-VGAE/tree/main/images/table.png" >
</p>

<p align="center">
<img align="center" src="https://github.com/nairouz/BELBO-VGAE/tree/main/images/pc.png">
</p>

### Qualitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/BELBO-VGAE/tree/main/images/vis.png">
</p>


## Usage

We provide the code of BELBO-VGAE. For each dataset, we provide the pretraining weights. The data is also provided with the code. Users can perform their own pretraining if they wish. For instance, to run the code of BELBO-VGAE on Cora, you should clone this repo and use the following command: 
```
python3 ./BELBO-VGAE/main_cora.py
```

## Built With

All the required libraries are provided in the ```requirement.txt``` file. The code is built with:

* Python 3.6
* Pytorch 1.7.0
* Scikit-learn
* Scipy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

* **Nairouz Mrabah** - *Grad Student (Université du Québec à Montréal)* 
* **Mohamed Bouguessa** - *Professor (Université du Québec à Montréal)*
* **Riadh Ksantini** - *Professor (University of Bahrain)*

 
## Citation
  
  ```
@inproceedings{mrabah2023beyond,
  title={Beyond The Evidence Lower Bound: Dual Variational Graph Auto-Encoders For Node Clustering},
  author={Mrabah, Nairouz and Bouguessa, Mohamed and Ksantini, Riadh},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
  pages={100--108},
  year={2023},
  organization={SIAM}
}
  ```




