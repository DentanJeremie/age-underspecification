# Robustness against distribution shifts and ambiguity

Author: Jérémie Dentan ([jeremie.dentan@polytechnique.org](mailto:jeremie.dentan@polytechnique.org))

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

You can also download the data from command lines, however the datasets are not hosted by the author, so it cannot be guaranteed that the link will remain available. To do so, run the four following commands from the `/data`directory:

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

- About 15min of prior CPU computation, including the extraction of the text
- About 2h45 of training when setting `src.classifier.age_classifier.PER_EPOCH_EVALUATION` to 0 to avoid doing predictions within the epochs.

## License and Disclaimer

You may use this software under the Apache 2.0 License. See LICENSE.
