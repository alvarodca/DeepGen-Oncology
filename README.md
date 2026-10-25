# Deep Generative Models for Single-Cell Oncology

Current project explores single-cell transcriptomics for personalised oncology. It explores different methods (scVI, PCA, 
LDVAE, MrVI, scANVI) to encode this information into a latent space.

The goal of this project is to assess how effectively deep generative latent spaces capture underlying transcriptional profiles and cell-type heterogeneity across breast cancer single cells while controlling for patient-level technical variation.

The main questions we want to address are:

a) Should patient be treated as a batch to remove? Malignant cells are
genuinely patient-specific, while immune cells are shared

b) Do the clusters correspond to known cell types and states?
Can we recover tumour-cell heterogeneity (e.g. the single-cell intrinsic subtypes of the paper)
without labels?

c) Build patient-level representations (cell-type composition, pooled latent
embeddings, or sample-level models such as MrVI) and predict clinical subtype.

Here we have to note that we only have 26 patients, therefore, train/test split should be carefully done

d) With respect to scANVI, use scANVI with partial labels; how many labelled cells are
needed? Transfer annotations to an unseen patient.

e)  Posterior predictive checks: does the model reproduce gene-level means,
variances and the fraction of zeros? What does the choice of likelihood change?

## Data

splits.parquet -> Some of our research questions are supervised questions, which is why we define a lookup table which defines exactly what training/test fold each patient belongs to across all models.
This is because we have to split across patients, not cells, as this would lead to data leakage.

This parquet file defines two different test/train splits. 

- fold_lopo (Leave One Patient Out):
25 patients are used for training and 1 for testing. Maximizes our training set, however, one single test split could lead to big inference variance, therefore results should not be fully trusted.

- fold_5 (Stratified Group-5 Fold CV)