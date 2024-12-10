## [Local Salient Location-aware Anomaly Mask Synthesis for Pulmonary Disease Anomaly Detection and Lesion Localization in CT Images] (Under Review)


### Abstract

Automated pulmonary anomaly detection using computed tomography (CT) examinations is important for the early warning of pulmonary diseases and can support clinical diagnosis and decision-making. Most training of existing \textcolor{blue}{pulmonary disease} detection and lesion segmentation models requires expert annotations, which is time-consuming and labour-intensive, and struggles to generalize across atypical diseases. In contrast, unsupervised anomaly detection alleviates the demand for dataset annotation and is more generalizable than supervised methods in detecting rare pathologies. However, due to the large distribution differences of CT scans in a volume and the high similarity between lesion and normal tissues, existing anomaly detection methods struggle to accurately localize small lesions, leading to a low anomaly detection rate. To alleviate these challenges, we propose a local salient location-aware anomaly mask generation and reconstruction framework for pulmonary disease anomaly detection and lesion localization. The framework consists of four components: (1) a Vector Quantised Variational AutoEncoder (VQVAE)-based reconstruction network that generates a codebook storing high-dimensional features; (2) a unsupervised feature statistics based anomaly feature synthesiser to synthesise features that match the realistic anomaly distribution by filtering salient features and interacting with the codebook; (3) a transformer-based feature classification network that identifies synthetic anomaly features; (4) a residual neighbourhood aggregation feature classification loss that mitigates network overfitting by penalising the classification loss of recoverable corrupted features. Our approach is based on two intuitions. First, generating synthetic anomalies in feature space is more effective due to the fact that lesions have different morphologies in image space and may not have much in common. Secondly, regions with salient features or high reconstruction errors in CT images tend to be similar to lesions and are more prone to synthesise abnormal features. The performance of the proposed method is validated on one in-house dataset containing 63,610 CT images with five lung diseases.  Experimental results show that compared to feature-based, synthesis-based and reconstruction-based methods, the proposed method is adaptable to CT images with four pneumonia types (COVID-19, bacteria, fungal, and mycoplasma) and one non-pneumonia (cancer) diseases} and achieves state-of-the-art performance in image-level anomaly detection and lesion localization.

### Dataset

Please put the root directory of your dataset into the folder ./data/test. The root directory contain the two subfolder now: normal and abnormal. The most convenient way is to follow the sample file structure, as follows:

```
|-- data
    |-- test
        |-- normal
        |-- abnormal
            |-- *.png


```
