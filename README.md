## Overview

This repository contains the code and resources for the paper Citation Classification for Behavioral Analysis of a Scientific Field. David Jurgens, Srijan Kumar, Raine Hoover, Dan McFarland, Dan Jurafsky. Transactions of the Association for Computational Linguistics (TACL). 2018

For full details, see the [Project Website](http://jurgens.people.si.umich.edu/citation-function/) which has links to all the data.

## Requirements

This project uses Python 2 and requires the following packages

```
pycorenlp==0.2.0
fuzzywuzzy
joblib
sklearn
ftfy==4.4.3
```

The program was developed using Stanford CoreNLP 3.6.0 .  Later versions may work but have not been tested.

If you're running the preprocessing steps from scratch, you'll need to have the Stanford CoreNLP server running on port 8999.  See [https://stanfordnlp.github.io/CoreNLP/corenlp-server.html] for instructions.

## Getting things running

The whole project consists of a series of scripts that convert and classify the ACL Anthology, all detailed in `code/run-pipeline.sh`.  This file should allow you to replicate the full set of experiments.  Approximate versions of the code to generate figures and tables for each part of the paper are found in the Jupyter notebooks in `analysis/`.  

## Contact

For general questions, contact the first author.  For code issues, please file an issue and we'll debug it from there.


## Citing 
```
  @article{jurgens2018citation,
           title={Citation Classification for Behavioral Analysis of a Scientific Field},
           author={Jurgens, David and Kumar, Srijan and  Hoover,Raine  and McFarland, Dan and Jurafsky, Dan },
           journal={Transactions of the Association of Computational Linguistics},
           year={2018}
  }
```
