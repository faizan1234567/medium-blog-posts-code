# MLOps with Weights & Baises
Weights & Baises helps developers build better models faster. Quickly track experiments, version and iterate on datasets, evaluate model performance, reproduce models, and manage your ML workflows end-to-end.
![alt text](https://github.com/faizan1234567/medium-blog-posts-code/blob/main/images/mlops.png)

Image by [Analytics Vidhya](https://www.analyticsvidhya.com/)

## Installation
create an environment in Anaconda
```bash
conda create -n wandb python=3.8.0
conda activate wandb
pip install --upgrade pip 
```
```bash
pip install -r requirements.txt
```

## Usage
```bash
python train.py --epochs 10 --experiments 5 --lr 1e-3 --batch 128
```
Once the training is complete you will see links to view your logs, click on the link to 
view your project runs
![alt text](https://github.com/faizan1234567/medium-blog-posts-code/blob/main/images/wandlogging.PNG)



## Acknowledgements
[1]. Weights & Baises
