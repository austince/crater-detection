# Crater Detection
**Austin Cawley-Edwards  
Stevens Institute of Technology  
PEP 336: Intro to Astrophysics
Prof. Vlad Lukic**


A basic crater detection implementation in __Python3__.

## Install
If you do not want do install, you can run with:
`$ python3 -m crater_detection.cli ...`

### With pip
 * Run `sudo pip3 install -e .[test]` to install a local, 'in place' copy.
 * Run `sudo pip3 uninstall crater_detection` to uninstall.

### With setup.py
* Run `python3 setup.py develop` to install a local, 'in place' copy.
* Run `python3 setup.py develop --uninstall` to uninstall this copy.

* Run `python3 setup.py install` to install a local immutable copy.  

## Requirements
This was developed using Conda, you probably don't have to but it's the easiest way to get this running.
* run `conda create --name <env> --file conda-reqs.txt` to create the conda environment.
* run `source activate <env>`
* run `pip install -r requirements.txt` to install python pip requirements.
* Open CV version > 3.1

## Running
The package exposes the command line interface `crater-detect` with functions for generating and detecting.

`crater-detect --help` will give you the full options, but here are some helpful ones:

### Generate
Creates a crater field with 25 craters, ranging from 10 to 30px in radius, seeded with the number 1, 
with an angle of 35 degrees, and saves it to `test.png`.

```bash
$ crater-detect generate --verbose --rand-seed 1 --num-craters 25 --max-rad 30 --min-rad 10 --angle 35 -o test.png
```

![](./outputs/final/test.png)

### Detect
Runs detection on generated `test.png`, logs the output, and saves the output picture to `output.png`.

```bash
$ crater-detect detect -i test.png --verbose -o output.png 
```

![](./outputs/final/output-test.png)