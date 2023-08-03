# xGW-GAT

This repository is the official implementation of `xGW-GAT`, an explainable, graph attention network for n-ary, transductive, classification tasks for functional brain connectomes and gait impairment severity. Our associated paper, **"An Explainable Geometric-Weighted Graph Attention Network for Identifying Functional Networks Associated with Gait Impairment"** has been accepted to MICCAI 2023 and is supported by the MICCAI 2023 STAR award. Check out our [preprint on arXiv](https://arxiv.org/abs/2307.13108) *(publication coming soon)*!

Our pipeline of three modules: 
1) A stratified, learning-based sample selection method leveraging Riemannian metrics for connectome similarity comparisons
2) An attention-based, brain network-oriented prediction model
3) An explanation generator for individual and global attention masks that highlight salient Regions of Interest (ROIs) in predicting gait impairment severity states.

<p align="center">
  <img width="809" alt="Screenshot 2023-08-01 at 12 19 32 PM" src="https://github.com/favour-nerrise/xGW-GAT/assets/7450249/b031c09c-fa79-431b-8ffa-a002ac78f7a9">
</p>

> **An Explainable Geometric-Weighted Graph Attention Network for Identifying Functional Networks Associated with Gait Impairment**
>
> [Favour Nerrise](mailto:fnerrise@stanford.edu)<sup>1</sup>, [Qingyu Zhao]()<sup>2</sup>, [Kathleen L. Poston]()<sup>3</sup>, [Kilian M. Pohl]()<sup>2</sup>, [Ehsan Adeli]()<sup>2</sup><br/>
> <sup>1</sup>Department of Electrical Engineering, Stanford University, Stanford, CA, USA<br/>
> <sup>2</sup>Department of Psychiatry and Behavioral Sciences, Stanford University, Stanford, CA<br/>
> <sup>3</sup>Dept. of Neurology and Neurological Sciences, Stanford University, Stanford, CA, USA<br/>
>
> **Abstract:** *One of the hallmark symptoms of Parkinson's Disease (PD) is the progressive loss of postural reflexes, which eventually leads to gait difficulties and balance problems. Identifying disruptions in brain function associated with gait impairment could be crucial in better understanding PD motor progression, thus advancing the development of more effective and personalized therapeutics. In this work, we present an explainable, geometric, weighted-graph attention neural network (xGW-GAT) to identify functional networks predictive of the progression of gait difficulties in individuals with PD. xGW-GAT predicts the multi-class gait impairment on the MDS Unified PD Rating Scale (MDS-UPDRS). Our computational- and data-efficient model represents functional connectomes as symmetric positive definite (SPD) matrices on a Riemannian manifold to explicitly encode pairwise interactions of entire connectomes, based on which we learn an attention mask yielding individual- and group-level explainability. Applied to our resting-state functional MRI (rs-fMRI) dataset of individuals with PD, xGW-GAT identifies functional connectivity patterns associated with gait impairment in PD and offers interpretable explanations of functional subnetworks associated with motor impairment. Our model successfully outperforms several existing methods while simultaneously revealing clinically-relevant connectivity patterns. The source code is available at [this https URL](https://arxiv.org/abs/2307.13108) .*

## Installation Instructions
- Download the ZIP folder or make a copy of this repository, e.g.  ```git clone https://github.com/favour-nerrise/xGW-GAT.git```. 

### Dependencies
This code was prepared using Python 3.10.4 and depends on the following packages:

* torch==2.0.1
* #PyG
* -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
* torch_geometric==2.3.1
* torch-cluster 
* torch-scatter  
* torch-sparse  
* torch-spline-conv  
* scikit-learn >= 0.24.1
* pymanopt==2.1.1
* numpy
* pandas==2.0.3
* scipy==1.11.1

See more details and install all required packages using ```pip install -r requirements.txt```. We recommend running all code and making installations in a virtual environment to prevent package conflicts. See [here](https://docs.python.org/3/library/venv.html) for more details on how to do so. 

## Getting Started
### Prepare Your Data 
* Extract functional correlation matrices from your chosen dataset, e.g. PPMI, save it as an ```.npy``` file, and place the dataset files in the ```./datasets/``` folder under the root folder. The saved matrix should be of shape ```(num_subjects, node_dim, node_dim)```.
* Save associated subject metrics, e.g. gait impairment severity score., as an ```.npy``` file and also place them in the ```./datasets/``` folder. The saved matrix should be of shape ```(num_subjects)```.
* Configure the ```brain_dataset.py``` and related files in the associated folder to correctly read in and process your dataset. Code has been provided for our use case of the ```PRIVATE``` dataset. 

## Calling the Model

```bash
python main.py --dataset=<name_of_dataset> --model_name=gatv2 --sample_selection --explain
```
The --explain argument is optional and triggers providing attention-based explanations of your model's predictions and saves related explanation data to the ```outputs/explanations/``` folder. 

## Configuration Options

Different configurations for the models and dataset can be specified in the ```main.py``` file, such as ```num_epochs```, ```num_classes```, and ```hidden_dim```.

## Hyperparameter Tuning
This pipeline was configured for hyperparameter optimization with [nni](https://github.com/microsoft/nni). Tuning configurations can be modified in the ```src/nni_configs.config.yml``` file. Using a Colab/Jupyter notebook, this can be done as follows: 
Create a free [ngrok.com](https://ngrok.com/) account and copy your *AuthToken* to be able to use the UI. Then run the following lines.
```
! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip # download ngrok and unzip it
! unzip ngrok-stable-linux-amd64.zip
```
```
! ./ngrok authtoken <AuthToken> 
```
```
! nnictl create --config src/nni_configs/config.yml --port 5000 &
```
```
get_ipython().system_raw('./ngrok http 5000 &') # get experiment id
```
```
! curl -s http://localhost:4040/api/tunnels # don't change the port number 4040
```
```
!nnictl stop <experiment_id> # stop running experiment
```


## Acknowledgments
This work was partially supported by NIH grants  (AA010723, NS115114, P30AG066515), Stanford School of Medicine Department of Psychiatry and Behavioral Sciences Jaswa Innovator Award, UST (a Stanford AI Lab alliance member), and the Stanford Institute for Human-Centered AI (HAI) Google Cloud credits.} FN is funded by the Stanford Graduate Fellowship and the Stanford NeuroTech Training Program Fellowship. 

This code was developed by Favour Nerrise (fnerrise@stanford.edu). We also thank [@Henny-Jie](https://github.com/HennyJie/), [@basiralab](https://github.com/basiralab/), and [@pyg-team](https://github.com/pyg-team/) for their related works and open-source code on [IBGNN](https://github.com/HennyJie/IBGNN) + [BrainGB](https://github.com/HennyJie/BrainGB), [RegGNN](https://github.com/basiralab/RegGNN/), and [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric), respectively, which served as great resources for developing our methods and codebase.


## Citation
Please cite our paper *(pre-print for now)* when using **xGW-GAT**:
```latex
@misc{nerrise2023explainable,
      title={An Explainable Geometric-Weighted Graph Attention Network for Identifying Functional Networks Associated with Gait Impairment}, 
      author={Favour Nerrise and Qingyu Zhao and Kathleen L. Poston and Kilian M. Pohl and Ehsan Adeli},
      year={2023},
      eprint={2307.13108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
