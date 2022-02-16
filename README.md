# Towards an Efficient ML System: Unveiling a Trade-off between Task Accuracy and Engineering Efficiency in a Large-scale Car Sharing Platform

Official Pytorch code to reproduce the main results in the paper.


## Prepare Dataset

TBD

## Training a Model 

To train a model, run:
1. Efficiency-Centric Model
    ```   
    sh run/pmg_uni.sh
    ```
2. Task-Centric Model
    ```   
    sh run/pmg_baseline_defect.sh
    sh run/pmg_baseline_dirt.sh
    ```
3. Multi-Task Model 
    ```   
    sh run/pmg_multi.sh
    ```
## Evaluation  

You can evaluate the trained model via jupyter notebook
1. Test accuracy 
    ```   
    test_sofar/*.ipynb
    ```
2. External validation
    ```   
    test_ext_val/*.ipynb
    ```
3. OOD rejection
    ```   
    test_ood/*.ipynb
    ```