# xGW-GAT

This repository is the official implementation of `xGW-GAT`, an explainable, graph attention network for n-ary, transductive, classification tasks for functional brain connectomes and gait impairment severity. Our associated paper, "An Explainable Geometric-Weighted Graph Attention Network for Identifying Functional Networks Associated with Gait Impairment" has been accepted to MICCAI 2023. 

This work was primarily developed by Favour Nerrise (fnerrise@stanford.edu). We also thank @Henny-Jie and Mehmet Arif Demirtaş for their inspiring, open-source works on IBGNN and RegGNN, respectively, which served as a great help in developing our codebase.

xGW-GAT framework consists of three modules: a stratified, learning-based sample selection method, an attention-based, brain network-oriented prediction model, and an explanation generator for individual and global masks that  highlight disorder-specific biomarkers, including important Regions of Interest (ROIs).
