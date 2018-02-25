1. The Python script (stopword_stem.py) does stemming and stopword removal to generate new file for the given file.

2. To run script.py, you will need to install some necessary Python packages:

 a. For linux based systems, makefile(make.sh) installs required packages, including nltk (which does the stemming and stop words removal).

 b. For mac, a script similar to make.sh should work. 

 c. For windows, follow the instructions from: http://www.nltk.org/install.html

Let us know if you run into problems installing these packages.

3. You should install the relevant packages first. On successful installation, run the Python script as:
python3 stopword_stem.py <old_file> <new_file>
