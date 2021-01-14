# Neural-EDU-Segmentation
A toolkit for segmenting Elementary Discourse Units (clauses).
We implement it as is described in our EMNLP paper: [Toward Fast and Accurate Neural Discourse Segmentation](http://www.aclweb.org/anthology/D18-1116)

# Docker-based installation and usage

```
docker build -f Dockerfile -t neural-edu-seg .

cat /tmp/neuralseg/input/input_short.txt 
Although they didn't like it, they accepted the offer.

docker run -v /tmp:/tmp -it neural-edu-seg /tmp/neuralseg/input/input_short.txt
2020-12-12 20:57:03.762424: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
/usr/local/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:97: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
/usr/local/lib/python3.6/site-packages/sklearn/utils/linear_assignment_.py:22: FutureWarning: The linear_assignment_ module is deprecated in 0.21 and will be removed from 0.23. Use scipy.optimize.linear_sum_assignment instead.
  FutureWarning)
Although they did n't like it ,
they accepted the offer .
```


### Requirements
- Python 3.5
- Tensorflow>=1.5.0
- allennlp>=0.4.2
- See `requirements.txt` for the full list of packages

### Data

We cannot provide the complete [RST-DT corpus](https://catalog.ldc.upenn.edu/products/LDC2002T07) due to the LDC copyright.
So we only put several samples in `./data/rst/` to test the our code and show the data structure.

If you want to train or evaluate our model on RST-DT, you need to download the data manually and put it in the same folder. Then run the following command to preprocess the data and create the vocabulary:

```
python run.py --prepare
```


### Evaluate the model on RST-DT:

We provide the vocabulary and a well-trained model in the `./data/` folder. You can evaluate the performance of this model after preparing the RST-DT data as mentioned above:

```
python run.py --evaluate --test_files ../data/rst/preprocessed/test/*.preprocessed
```

The performance of current model should be as follows:
```
'precision': 0.9176470588235294, 'recall': 0.975, 'f1': 0.9454545454545454}
```

Note that this is slightly better than the results we reported in the paper, since we re-trained the model and there is some randomness here.

### Train a new model

You can use the following command to train the model from scratch:

```
python run.py --train
```

Hyper-parameters and other training settings can be modified in `config.py`.

### Segmenting raw text into EDUs

You can segment files with raw text into EDUs:

```
python run.py --segment --input_files ../data/rst/TRAINING/wsj_110*.out --result_dir ../data/results/
```

The segmented result for each file will be saved to the `--result_dir` folder with the same name. Each EDU is written as a line.


### Citation

Please cite the following paper if you use this toolkit in your work:

```
@inproceedings{wang2018edu,
  title={Toward Fast and Accurate Neural Discourse Segmentation},
  author={Wang, Yizhong and Li, Sujian and Yang, Jingfeng},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={962--967},
  year={2018}
}
```
