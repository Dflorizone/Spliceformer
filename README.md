# Spliceformer: Transformer-Based Splice Site Prediction [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14019451.svg)](https://zenodo.org/badge/DOI/10.5281/zenodo.14019451.svg)

This project is forked from the Spliceformer repository (https://github.com/benniatli/Spliceformer), and includes some additional scripts and notebooks used to test Transformer-40k and SpliceAI-10k. 

The scripts and notebooks I developed were used to reproduce some of Spliceformer's results, test Transformer-40k on the mouse genome, and try and apply some explainability techniques to both SpliceAI-10k and Transformer-40k. The scripts and notebooks I used are found within the Code directory, and the unused notebooks from the original Spliceformer project are located within the additional_notebooks folder.

Additional details on the original Spliceformer project can be found in the [repository](https://github.com/benniatli/Spliceformer), and in the [paper](https://www.nature.com/articles/s42003-024-07298-9).

## Scripts and Notebooks
Here is a summary of what the scripts and notebooks were used for:

1. Construct datasets
    * construct_ENSMBL_datasets.ipynb
    * construct_GENCODE_datasets.ipynb
    * construct_ENSEMBL_mouse.ipynb

2. Model testing
    * ensembl_test.py
    * gencode_test.py
    * spliceAI_ensembl_test.py
    * mouse_ensembl_test.py
    * run_ensembl_test.sh
    * run_ensembl_test_spliceai.sh
    * run_gencode_test.sh
    * run_mouse_ensembl_test.sh
    * transformer_vs_spliceai.ipynb
    * mouse_evaluation.ipynb
    * gencode_test.ipynb
    * miscellaneous_plots.ipynb

3. Explainability
    * get_attention_plots.ipynb (used to visualize attention scores)
    * intronless_interpretation.ipynb (used to analyze intronless transcripts)
    * CNN_interpretability.ipynb (used to implement Grad-CAM)
    * grad_cam_seq_visualization.ipynb (used to evaluate Grad-CAM results with MEME)
    * deepshap.ipynb (used to implement DeepSHAP)

4. Model defined in src
    * model_gradcam.py (modifying original model.py slightly to work with Grad-CAM)
    * model_shap.py (modifying original model_shap.py slightly to work with DeepSHAP)


