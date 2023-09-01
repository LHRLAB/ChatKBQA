# ChatKBQA

##  General Setup 

### Environment Setup
```
conda create -n chatkbqa python=3.8
conda activate chatkbqa
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt
```

###  Freebase KG Setup

Below steps are according to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup). 
#### How to install virtuoso backend for Freebase KG.

1. Clone from `git@github.com:dki-lab/Freebase-Setup.git`:
```
cd Freebase-Setup
```

2. The latest offical data dump of [Freebase](https://developers.google.com/freebase) can be downloaded. However, in the official dump, the format of some literal types is not fully compatible with the N-Triples RDF standard (it's missing type decoration such as `^^<http://www.w3.org/2001/XMLSchema#integer>`), which may cause it to fail to load into triplestores like Virtuoso. We fixed this issue. Our processed Virtuoso DB file can be downloaded from [here](https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip) or via wget (WARNING: 53G+ disk space is needed):
```
tar -zxvf virtuoso_db.zip
```

In case you'd like to load your own RDF into Virtuoso, see [here](http://vos.openlinksw.com/owiki/wiki/VOS/VirtBulkRDFLoader) for instructions. If you prefer some other triplestore, try fixing the literal format issue of Freebase using the script `fix_freebase_literal_format.py` to get the N-Triples-formatted data. 

3. Managing the Virtuoso service

We provide a wrapper script (`virtuoso.py`, adapted from [Sempre](https://github.com/percyliang/sempre)) for managing the Virtuoso service. To use it, first change the `virtuosoPath` in the script to your local Virtuoso directory. Assuming the Virtuoso db file is located in a directory named `virtuoso_db` under the same directory as the script `virtuoso.py` and 3001 is the intended HTTP port for the service, to start the Virtuoso service:

First, change 11st line of `virtuoso.py` to ```virtuosoPath = "../virtuoso-opensource"```.

Then,
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

and to stop a currently running service at the same port:

```
python3 virtuoso.py stop 3001
```

A server with at least 100 GB RAM is recommended. You may adjust the maximum amount of RAM the service may use and other configurations via the provided script.
