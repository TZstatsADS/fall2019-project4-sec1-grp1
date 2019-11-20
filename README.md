# Project 4: Algorithm implementation and evaluation: Collaborative Filtering

### [Project Description](doc/project4_desc.md)

Term: Fall 2019

+ Team 1
+ Projec title: Collaborative Filitering By Different Regularization
+ Team members
	+ Cao, Rui rc3208@columbia.edu

	+ Qiu, Feng fq2150@columbia.edu

 	+ Yao, Nichole yy2860@columbia.edu

	+ Ye, Xuanhong xy2387@columbia.edu
	
	+ Ponkshe, Tushar tvp2110@columbia.edu
	
+ Project summary: The goal of this project is to evaluate different recommendation approaches. We implemented Matrix Factorization nonprobabilistic algorithm with SGD approach and postprocessing KNN, and three regularizations: penalty magnitudes, bias and intercepts and temporal dymanics. And we applied MovieLens datasets (http://movielens.org) For evaluation part, we compared the performance for these different algorithms by RMSE for Movie dataset.

+ The report is formed by the Jupyter Notebook in `.ipynb` file: [main.ipynb](doc/main.ipynb).

+ The functions used in the whole report can be find [here](lib/Matrix_Factorization_A1.py).

	
**Contribution statement**: 

+ Function Implementation:
  + Uniform SGD: Rui Cao
  + SGD adding bias: Rui Cao 
  + SGD adding temporal dynamics: Qiu, Feng
  + Postprocessing: Xuanhong Ye
  
+ Evaluation:
  + Parameter Tuning: Yao, Nichole
  + Report: Yao, Nichole
  
+ Presentation: Yao, Nichole

+ Github management: Rui Cao, Qiu Feng

note: Ponkshe Tushar doesn't participate to the project. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
|-- lib/
|-- data/
|-- doc/
|-- figs/
|-- output/
```

Please see each subfolder for a README file.
