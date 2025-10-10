# Physics-Informed-Neural-Networks-for-Wireless-Channel-Estimation-with-Limited-Pilot-Signals

Official implementation of **" Physics Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals"**.

## 📋 Abstract

Accurate wireless channel estimation is critical for next-generation communication systems, yet traditional model-based methods struggle in complex environments with limited pilot signals, while purely data-driven approaches lack physical interpretability and require extensive pilot overhead. This paper presents a novel PINN framework that synergistically combines model-based channel estimation with environmental propagation characteristics to achieve superior performance under pilot-constrained scenarios. The proposed approach employs an enhanced U-Net architecture with transformer modules and cross-attention mechanisms to fuse initial channel estimates with RSS maps. Comprehensive evaluation using realistic ray-tracing data from urban environments demonstrates significant performance improvements, achieving over 5 dB gain in NMSE compared to state-of-the-art methods, with particularly strong performance in pilot-limited scenarios (achieving around -13 dB NMSE with only four pilots at SNR = 0 dB). The proposed framework maintains practical computational complexity, making it viable for massive MIMO systems in upper-mid band frequencies. Unlike black-box neural approaches, the physics-informed design provides a more interpretable channel estimation method.


## Dataset

There is a dataset of two environments and two different frequency bands in the upper mid-band simulated in Wireless Insite. Channel information, where each channel is associated with one array position and contains 25 paths. You can use <em> make_correct_channels.py </em> to make the channel tensors according to the paper for further evaluations. 

## Neural Network

The Python codes for the network design and training are in <em>Model.py</em> and <em>train.py</em>, respectively. If you are using other datasets, you might need to adjust the parameters accordingly. 

