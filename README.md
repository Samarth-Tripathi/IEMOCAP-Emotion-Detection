# IEMOCAP-Emotion-Detection
Multi-modal Emotion detection from IEMOCAP on Speech, Text, Motion-Capture Data using Neural Nets.



We attempt to exploit this effectiveness of Neural networks to enable us to perform multimodal Emotion recognition on IEMOCAP dataset using data from Speech, Text, and Motion capture data from face expressions, rotation and hand move- ments. Prior research has concentrated on Emotion detection from Speech on the IEMOCAP dataset, but our approach is the first that uses the multiple modes of data offered by IEMOCAP for a more robust and accurate emotion detection.


For this we explore various deep learning based architectures to first get the best individual detection accuracy from each of the different modes. We then try combine them in an ensemble based architecture to allow for end-to-end training across the dif- ferent modalities using the variations of the better individual models. Our ensemble consists of Long Short Term Memory networks, Convolution Neural Networks, Fully connected Multi-Layer Perceptrons and we complement them using techniques such as Dropout, adaptive optimizers such as Adam, pretrained word-embedding models and Attention based RNN decoders. Comparing our speech based emotion detection with [13] we achieve 62.72% accuracy compared to their 62.85%; and comparing with [17] we achieve 55.65% accuracy compared to their CTC based 54% ac- curacy. After combining Speech (individually 55.65% ac- curacy) and Text (individually 64.78% accuracy) modes we achieve an improvement to 68.40% accuracy. When we also account MoCap data (individually 51.11% accuracy) we also achieve a further improvement to 71.04%.

Please refer to the pdf and ppt for more details.
