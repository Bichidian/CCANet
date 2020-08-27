This code repository is for the paper:

> Pehlevan, C., Zhao, X., Sengupta, A. M., & Chklovskii, D. (2020). Neurons as Canonical Correlation Analyzers. Frontiers in computational neuroscience, 14, 55.



Since the MNIST dataset is loaded using old tensorflow API, an old tensorflow=1.x version is required.

An external MATLAB script is used, requiring MATLAB and its interface with python installed. The script is taken from https://www.dropbox.com/sh/dkz4zgkevfyzif3/AABK9JlUvIUYtHvLPCBXLlpha?dl=0, which is the code for the paper:

> Arora, R., Marinov, T. V., Mianjy, P., & Srebro, N. (2017). Stochastic approximation for canonical correlation analysis. In Advances in Neural Information Processing Systems (pp. 4775-4784).



To reproduce the results, first produce data files by running CCACompare.py and CCADendrites.py, which will take some time. Then run CCAPublicationFigure.py to produce the figures.