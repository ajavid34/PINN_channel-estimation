# RSS map-assisted MIMO channel estimation in the upper mid-band under pilot constraints

Official implementation of **" RSS map-assisted MIMO channel estimation in the upper mid-band under pilot constraints"**. 


## 📋 Abstract

Accurate wireless channel estimation is critical for next-generation wireless systems, enabling precise precoding for effective user separation, reduced interference across cells, and high-resolution sensing, among other benefits. 
Traditional model-based channel estimation methods suffer, however, from performance degradation in complex environments with a limited number of pilots, while purely data-driven approaches lack physical interpretability, require extensive data collection, and are usually site-specific. This paper presents a novel PINN framework that synergistically combines model-based channel estimation with a deep network to exploit prior information about environmental propagation characteristics and achieve superior performance under pilot-constrained scenarios. The proposed approach employs an enhanced U-Net architecture with transformer modules and cross-attention mechanisms to fuse initial channel estimates with RSS maps to provide refined channel estimates. Comprehensive evaluation using realistic ray-tracing data from urban environments demonstrates significant performance improvements, achieving over 5 dB gain in NMSE compared to state-of-the-art methods, with particularly strong performance in pilot-limited scenarios (achieving around -13 dB NMSE with only four pilots at SNR = 0 dB) and robustness across different frequencies and environments with only minimal fine-tuning. We further extend the decoder for multi-step temporal prediction, enabling accurate forecasting of several future channel snapshots from a single estimate—useful for proactive beamforming and scheduling in mobile scenarios. The proposed framework maintains practical computational complexity, making it viable for massive MIMO systems in upper-mid band frequencies. Unlike black-box neural approaches, the physics-informed design provides a more interpretable channel estimation method.


## Dataset

There is a dataset of two environments and two different frequency bands in the upper mid-band simulated in Wireless Insite. Channel information, where each channel is associated with one array position and contains 25 paths. You can use <em> make_correct_channels.py </em> to make the channel tensors according to the paper for further evaluations. 

## Neural Network

The Python codes for the network design and training are in <em>Model.py</em> and <em>train.py</em>, respectively. If you are using other datasets, you might need to adjust the parameters accordingly. 

