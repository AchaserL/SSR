## Dataset
We use DBP15k in our experiments. DBP15k can be downloaded from [JAPE](https://github.com/nju-websoft/JAPE)

### Dependencies
* Python 3
* Tensorflow 2.0
* Scipy
* Numpy
* Pandas
* Scikit-learn

### Running
For example, to run AlN-SSR on DBP15k ZH-EN, use the following script (supposed that the DBP15k data has been downloaded into the folder '../data/'):
```python
python3 main.py --input ../data/DBP15k/zh_en/mtranse/0_3/
```
