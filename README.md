# Using Error Level Analysis to remove Underspecification

Author: Jérémie Dentan

<p align="center">
  <img src="doc/figures/teaser.jpg"
  alt="Teaser"
  width=100%/>
</p>

## Overview

This repository contains the implementation of the work of the author for a data competition. The goal of this competition is to predict the age Young/Old based on pictures of people, in the presence of underspecification. The underspecification comes from the fact that a text is written on the images, wich is sometimes correlated to the true label, and sometimes not.

For detailed presentation of the competition and the task, please refer to:

- Our technical report, available at [http://dx.doi.org/10.13140/RG.2.2.25127.21925](http://dx.doi.org/10.13140/RG.2.2.25127.21925)
- Our presentation, available at [/doc/defense.pdf](/doc/defense.pdf)

## Set up

The code provided here is expected to run under **Python 3.9** with the PYTHONPATH set to the root of the project. To do so, you should run the following from the root of the project:

```bash
export PYTHONPATH=$(pwd)
pip install -r requirements.txt
```

### Downloading the data

Before running the code, you should download the data and put the `human_age_shared` and `human_hair` folders in the `/data` directory that is at the root of the project. The `zip` file of those folders are available here:

- File `human_age_shared.zip`, that you should unzip, creating the folder `/data/human_age_shared` containing, among others, `y_labeled.csv`: [https://www.icloud.com/iclouddrive/02aZwmJLNKKQ20m3M39txhYjQ#human%5Fage%5Fshared](https://www.icloud.com/iclouddrive/02aZwmJLNKKQ20m3M39txhYjQ#human%5Fage%5Fshared)
- File `human_hair.zip`, that you should unzip, creating the folder `/data/human_hair` containing, among others, `y_labeled.csv`: [https://www.icloud.com/iclouddrive/069iFOZ6-V6LSCzYGaqiYyoGQ#human%5Fhair](https://www.icloud.com/iclouddrive/069iFOZ6-V6LSCzYGaqiYyoGQ#human%5Fhair)

You can also download the data using command lines, however the datasets are not hosted by the author, so it cannot be guaranteed that the link will remain available. To do so, run the four following commands from the `/data` directory:

```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17pzoXHnPJwxyKaNEF7JdwkkVwW0IotN-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17pzoXHnPJwxyKaNEF7JdwkkVwW0IotN-" -O human_age_shared.zip && rm -rf /tmp/cookies.txt

unzip -q  'human_age_shared.zip'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12VpKhiDVXCMiiak90wpPxF_r6nT_LPC0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12VpKhiDVXCMiiak90wpPxF_r6nT_LPC0" -O human_hair.zip && rm -rf /tmp/cookies.txt

unzip -q  'human_hair.zip'
```

## Run the code

### Execution

Once you have done the setup, you can run the following to reproduce our experiments. This will create the predictions in `/output` as well as log files in `/logs` describing what have been going on:

```bash
python -m src.classifier
```

### Computation time

Using a NVIDIA GeForce RTX 3090 24Go for the GPU computation and a  Intel Xeon W-1290P 3.70GHz with 10 physical core for the CPU computations, the pipeline takes:

- About 20min of prior CPU computation, including the extraction of the text
- About 2h45 of training when setting `src.classifier.age_classifier.PER_EPOCH_EVALUATION` to 0 to avoid doing predictions within the epochs. If this variable is set to 1, it adds about 5min per epoch.

## License and Disclaimer

You may use this software under the Apache 2.0 License. See LICENSE.
