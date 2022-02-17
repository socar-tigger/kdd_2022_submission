# Towards an Efficient ML System: Unveiling a Trade-off between Task Accuracy and Engineering Efficiency in a Large-scale Car Sharing Platform

Official Pytorch code to reproduce the main results in the paper.

## Paper Abstract

Upon the significant performance of the supervised deep neural networks, conventional procedures of developing ML system are task-centric, which aims to maximize the task accuracy. However, we scrutinized this task-centric ML system lacks in engineering efficiency when the ML practitioners solve multiple tasks in their domain. To resolve this problem, we propose an efficiency-centric ML system that concatenates numerous datasets, classifiers, out-of-distribution detectors, and prediction tables existing in the practitioners' domain into a single ML pipeline. Under various image recognition tasks in the real world car-sharing platform, our study illustrates how we established the proposed system and lessons learned from this journey as follows. First, the proposed ML system accomplishes supreme engineering efficiency while achieving a competitive task accuracy. Moreover, compared to the task-centric paradigm, we discovered that the efficiency-centric ML system yields satisfactory prediction results on multi-labelable samples, which frequently exist in the real world. We analyze these benefits derived from the representation power, which learned broader label spaces from the concatenated dataset. Last but not least, our study elaborated how we deployed this efficiency-centric ML system is deployed in the real world live cloud environment. Based on the proposed analogies, we highly expect that ML practitioners can utilize our study to elevate engineering efficiency in their domain.


## Prepare Dataset

Please visit the Github Repository (https://github.com/socar-kp/sofar_image_dataset) to check sample images utilized in this paper or acquire a full access to the dataset.

If you aim to reproduce the study, we recommend you to submit a request form to the dataset in the aforementioned Github Repository.

In case of any problems or inquiries, please raise the Issue.


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
