# Reproducing Experiment Results
The experiment script and plotting functions are run as two separate processes. Run the bash scripts first to generate log files, then run the plot files on the logs.
This experiment assumes Python and Pip are installed.

## Hardware
We used a baremetal Chameleon Cloud instance which ran on Ubuntu 20.04. Most Linux operating systems will be sufficient as long as they allow access to Perf.

## Running the Bash Scripts
This experiment requires [Perf](https://perfwiki.github.io/main/), so the bash scripts must be run as root. Therefore, all installations for these scripts should also be in root. Start a root terminal with `sudo -s`.
Install perf with the following instructions:
```
sudo apt update
sudo apt upgrade
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
```
Then install all other requirements with `pip install -r requirements.txt`.

The experiments are run on YOLOv7. First, clone YOLOv7 from [WongKinYiu's repository](https://github.com/WongKinYiu/yolov7) into this directory with `git clone https://github.com/WongKinYiu/yolov7`.

In the yolov7 directory run `pip install -r requirements.txt`.

Copy the experiment bash scripts into the yolov7 directory:
```
cp ../exp1.bash ./
cp ../exp2.bash ./
```

Then run both bash scripts. Stop all other processes running and run the bash scripts at separate times to ensure accurate energy measurements.

## Making the Plots
The plotting scripts use the log files to generate the results of the experiments. They require a graphical display, which will not be automatically setup in a cloud machine. The user may choose to use graphical forwarding or send the log files and plotting scripts to a machine which has a physical display. The plotting scripts have been testing on Linux, but likely will run on other operating systems as well.

* plot_exp1.py - Figures 1 & 2
* plot_exp2.py - Figures 3 & 4

In the terminal where the plotting scripts will be run, install all other requirements with `pip install -r requirements.txt`.

Move the resulting log files from the bash scripts to the `out` directory and change the path in the Python scripts to point to the correct log files.

Run the scripts:
```
python3 plot_exp1.py
python3 plot_exp2.py
```

When results are successfully reproduced, figures will display that show a similar relationship to the figures in the README in the top-level directory.
