# Pytorch Implementation of SN-GAN with CIFAR10

## Requirements
nsml (not neccessary)
visdom

## Paper
[Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-)

## Run Example
**If you have nsml:**
```{python}
nsml run -d cifar10_python -a "--sn"                   # Spectralnorm
nsml run -d cifar10_python                             # No SN
nsml run -d cifar10_python -a "--sn --inception_score" # Calculate Inception score
```
**If you don't:**
```{python}
python main.py --sn                    # Spectralnorm
python main.py                         # No SN
python main.py --sn --inception_score" # Calculate Inception score
```

## Architecture

GAN Architecture is adopted from the papers' Appendix B.4 for CIFAR10
<p align="center">
  <img src="./assets/architecture.png">
</p>

## Results

*Generated images*
<p align="center">
  <img src="./assets/gen_example.png">
</p>

*Loss*
<p align="center">
  <img src="./assets/g_loss.png">
</p>
<p align="center">
  <img src="./assets/d_loss.png">
</p>
<p align="center">
  <img src="./assets/d_loss_fake.png">
</p>
<p align="center">
  <img src="./assets/d_loss_real.png">
</p>


