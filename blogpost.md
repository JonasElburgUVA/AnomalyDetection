## Introduction

With rapid improvements in generative AI, it is becoming more difficult to distinguish real from generated data (e.g. deepfakes). Therefore, the need to recognize generated works became more critical. This problem can be posed as an anomaly detection task, with the goal of identifying artifacts that typically occur in modified or generated images but not in real ones. Marimont & Tarroni (2021) recently introduced an anomaly detection model yielding promising results in the field of medical imaging. In this blog post, we will investigate the performance of this model for the purpose of detecting generated and artificially altered images of faces, or *fakes<sup>1</sup>*. First, we will discuss the methodology used by Marimont & Tarroni (2021). Then we will explain how this method is applied to distinguishing modified/generated faces from real faces. Finally, we present the results of this method on various datasets and evaluate its robustness to different face manipulation techniques.

> <sup>1</sup> We considered completely generated as well as partially altered images of people’s faces in this research. We will refer to all of these as fakes in this blog post for simplicity.

### Anomaly Detection through Latent Space Restoration using Vector Quantized Variational Auto-Encoders

In response to the 2020 Medical Out of Distribution (MOOD) challenge, Marimont & Tarroni (2021) developed a novel method for detecting anomalies in image data. A Vector-Quantized Variational Auto-Encoder (VQ-VAE), as introduced by van den Oord et al. (2017), was trained on two datasets of medical images (brain and abdominal scans). The main difference between a VQ-VAE and a regular VAE is the latent space, made of a finite set of discrete vectors rather than being continuous. The technical details and (dis)-advantages of using VQ-VAEs will be discussed in Methods > VQ-VAE. The latent space of the VQ-VAE was regularized by a learned autoregressive prior using the PixelSNAIL architecture (Chen et al., 2018). This architecture allows the calculation of both sample and pixel-wise anomaly scores. Since the MOOD challenge test data is withheld, the model was evaluated using a small toy dataset with clear artificial anomalies (figure 1). On this toy dataset, the VQ-VAE outperformed a regular VAE model in detecting anomalies on the sample and pixel levels.

<p
   style="text-align: center">
    <img src="/uploads/upload_32234ee2e916356263b3ce2bcc30dc5e.png" height="250">
    <center>Figure 1: Example of a slice containing an artificial anomaly.</center>
</p>


The method performed relatively well on the withheld test data, resulting in the second-best performance of the challenge (Zimmerer et al., 2020). However, it was noted that performance is especially high for obvious abnormal inputs (e.g., containing large hyperintense patches originating from a tumor), while performance on images with more subtle anomalies is much less stable, limiting the practical use of these models. The MRI brain scan dataset was obtained from healthy young individuals, meaning that the brains are quite uniform between subjects. The other dataset consisting of abdominal CT scans, however, was obtained from both healthy and unhealthy participants of varying age groups. Furthermore, more variance is expected since multiple organs are contained within the abdomen. Performance seems sensitive to this variance since the model performed much better on the MRI dataset than on the high-variance dataset containing healthy and unhealthy abdominal CT scans (Marimont \& Tarroni, 2021; Zimmerer et al., 2020). It is unclear whether this results from an inherent limitation of this model or whether this could be resolved with more data.

## Anomaly Detection in Fake Image Recognition

The increase in the number of automatic facial recognition systems (Kortli et al., 2020; Tolba et al., 2006) stimulated the development of different techniques to fool such models (e.g. Parmar et al., 2022; Xue et a., 2019). Several techniques, including VAE-based models, have been developed to detect manipulated faces with some success. However, VQ-VAE models have not yet been applied to this task. These models are relatively cheap to train, can explain their classification decisions using residuals, and have been shown to outperform regular VAEs in anomaly detection (Marimont & Tarroni, 2021). We will evaluate VQ-VAE based anomaly detection applied to morphed faces.

Our contribution can be split into three parts, since the results have implications for both the field of medical anomaly detection and fake face detection:

- **Firstly**, our results investigate whether the model can perform well on high-variance data (such as images of faces) when larger datasets are available.
- **Secondly**, our results will show whether this approach can be used for detecting manipulated faces in an effort to support the development of the models which are able to do so.
- **Thirdly**, we improved the training time of the code written by Marimont & Tarroni (2021) by making improvements in the data preprocessing pipelines.

> We can write more clearly about our contributions once we have results
> Maybe this could simply be a list instead of having firstly, secondly, thirdly?

## Methods

### Vector-Quantized Auto-Encoder

Vector Quantized Auto-Encoders (VQ-VAEs; Van den Oord et al., 2017), similarly to regular VAEs, consist of an encoder, a latent space, and a decoder. The encoder transforms input data into a lower dimensional latent representation, and the decoder transforms this latent representation back to its original dimensions. The defining trait of a VQ-VAE, is that a quantization module follows the encoder to make the latent space discrete. In this quantization, each initial output of the encoder is mapped to one of $K$ vectors that make up the latent space. These $K$ vectors and their $D$-dimensional entries are referred to as the codebook. The mapping is done by replacing each $D$-dimensional vector $z_e(x)$ in the encoder's output $Z(x)$, with the nearest vector in the codebook, $v_k$, based on a nearest neighbor search (see the equation below). This means that, with a $K \times D$ sized codebook, the encoder output before quantization should be of the shape $H \times W \times D$. The shape will be the same after quantization, but the entries of the $H \times W$ vectors will be replaced by the nearest codebook vectors. This quantized tensor, $Z_q(x)$, functions as the input to the decoder.

$$
z_e(x) = v_k \text{ where } k = \text{argmin}_j || z_e(x) - v_j ||_2
$$

To ensure expressiveness and flexibility of the discrete latent space, the entries of the codebook are learned during training. This means that, aside from the regular VAE loss, another term is added to the learning objective. This “codebook loss”, derived from the average distance between the encoder output and the nearest vector in the codebook, enables us to learn a distribution over discrete vectors in the latent space. Finally, a penalty is introduced for very large valued encoder outputs, to ensure the latent space does not grow unnecessarily. This is called “commitment loss”, and corresponds to the third term in the equation below. The full learning objective is then given by the following equation:
 

![](/uploads/upload_6bd29791c51f4eb4dfcd4efc188b3dc7.png)


![](/uploads/upload_006277838d72473a0906bc0ff2b3da28.png)

> The architecture of the VQ-VAE (van den Oord, et al. 2017). Note that currently we have swapped $e$ with $v$ in our calculations

### Auto-Regressive model as the anomaly detector

Crucial for our purpose, a probabilistic distribution over the latent space is learned. This is referred to as the (learned) prior. This is essential for making the VQ-VAE generative, since it allows us to sample new data-points in the latent space. In our model, the latent distribution is learned using PixelSNAIL (Chen et al., 2018). PixelSNAIL is an autoregressive model, meaning that it assigns probabilities based on all previous data-points. Generally, this means that $p(x) = \prod^{N}_{i} p(x_i|x_{i'<i})$. If, during inference, we quantize a vector $z_{e_{n}}(x)$ to the vector $z_{q_n}(x)$, but the joint probability of the quantized vectors up to the $n$th vector, given by $\prod^{n}_{i}p\left(z_{q_i}|z_{q_{i'<i}}\right)$, is low, we can infer that the input data is unlikely to be something the model is used to reconstructing. In other words, if we have trained on a large amount of non-anomalous data points, a low assigned probability suggests the presence of an anomaly. Since the latent space is difficult to interpret, we do not derive anomaly scores directly from these probabilities. Instead, when a highly unlikely configuration of latent codes occurs, we resample the last quantized code $z_x$ from the latent distribution using the autoregressive prior. Since we will now have a much more likely and expected configuration of latent codes, the decoded image should not contain anomalies anymore. We can thus use the difference between the input and output image to calculate anomaly scores.

![](/uploads/upload_61899469f62e78e30fa7fb8f4ee56f80.png)

The VQ-VAE architecture with a learned autoregressive prior comes with some advantages over a regular VAE.

**Pros**

- The learned prior distribution requires less assumptions than the typical gaussian prior used in VAEs. This makes the model less prone to bias and convergence issues.
- Since the prior is learned, the latent space is better at efficiently representing the data in lower dimensions. This results in better quality samples than regular VAEs.
- Thanks to the discrete latent space, the probabilities of anomalous vectors assigned by the AR model are more interpretable.

**Cons**

- The learning of the prior is computationally intensive
- More parameters, may require more training (but it’s a trade-off with the improved efficiency of training)

## Anomaly scores
When predicting whether an image is fake or not, we would like our model to not only accurately reflect the probability that an image is fake, but also what parts of the picture have been changed. The VQ-VAE model allows us to calculate an anomaly score for a full image (sample-wise) as well as for individual pixels (pixel-wise). The pixel-wise anomaly score can thus provide an explanation for the model’s sample-wise prediction. We will show how these scores are calculated.

### Sample-wise

> I would replace all the unaltered, non-anomalous etc ways we refer to normal data with "regular"

Given the output of the AR model and a threshold, the sample-wise anomaly score shows the likelihood of an input image being artificially generated or edited in some way. The metric is based on the finding that if the VQ-VAE has a large enough latent space, (meaning that it will learn the distribution of unaltered data quite well) it will be able to reconstruct the image so it does not include the anomaly. Additionally, the paper finds that the latent variables are different for normal and abnormal regions, resulting in the AR model assigning a low probability for the latter (meaning that the AR model predicts that there is a low probability that some region is from the prior distribution). The paper assigns a negative log-likelihood (NLL) threshold $\lambda_s$ that defines the highly unlikely latent variables. The final score will then be calculated by summing over the NLL of the latent variables above a threshold $\lambda_s$, over a total of $N$ variables.

$$
\begin{align}
    \text{AS}_\text{sample} = \sum_i^N \xi(p(x_i)) \\
    \xi(z) = \begin{cases}
        -\text{log}(z) & \text{if} -\text{log}(z) > \lambda_s \\
        0 & \text{otherwise}
    \end{cases}
\end{align}
$$

### Pixel-wise

> I would remove "(variables that are predicted to belong to the learned prior distribution)"

The pixel-wise score allows us to localize the anomalies in an images. It works using the restoration process introduced in (Chen et al., 2020), which can be broken down in the following steps:

1) Replace latent variables with a high loss using samples from the prior learned by the AR model, while keeping low loss latent variables (variables that are predicted to belong to the learned prior distribution) unaltered
2) Draw a new sample if its latent NLL is above a threshold $\lambda_p$.
3) Generate the restored image using the decoder
4) Compute the residual $|X - \text{Restoration}|$

For each image, multiple restorations are generated to reduce variance in the anomaly estimation. The restorations are then combined using a weighting factor $w_j$ defined as:

$$
w_j = \text{softmax}\left(\frac{k}{\sum_i^P |Y^i - X^i_j|}\right)
$$

$k$ is a softmax temperature parameter and the sum in the denominator is taken over all image pixels $P$. The $w$ will then reduce the weight of restorations which have lost consistency.

The final equation to get the score iw the weighted mean of all residuals:

$$
\text{AS}_\text{pixel} = \sum_j^S w_j |Y - X_j|
$$

As the final step, the $\text{AS}_\text{pixel}$ scores are run through a 3x3 MinPooling filter followed by a 7x7 AveragePooling filter to smooth the values.

## Datasets

### Training:

- [Flickr Faces HQ dataset](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00022.png)
    - The dataset consists of images of faces in canonical form.
    - Due to space constraints we experiment with a subset of the data containing 52k images.
    - Images are downsampled to 512 x 512 pixels.

<p style="text-align: center">
    <img src="/uploads/upload_783cc87a4c8be32db865dc6fcbb8d39a.png" alt>
    <center>Six images from the Flickr Faces HQ Dataset (FFHQ).</center>
</p>

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
    - The dataset consists of youtube videos, but the authors provide a script to extract video frames.
    - The original video is provided, alongside morphed videos using various DeepFake techniques.
    - The dataset is not public, but can be accessed by filling in [this form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) 

<p style="text-align: center">
    <img src="/uploads/upload_ec4f3eab8eb9d110284fff6bc3a4a7be.png" alt>
    <center>The original video.</center>
</p>

<p style="text-align: center">
    <img src="/uploads/upload_7667e5f0ea2c281c9f9a03efe9808f63.png" alt>
    <center>Manipulated "DeepFakes" video, created using the <a href="https://github.com/deepfakes/faceswap">FaceSwap Github repository</a>.</center>
</p>

<p style="text-align: center">
    <img src="/uploads/upload_5e83e0238a4a858e5992cc869f487b7a.png" alt>
    <center>Manipulated video using the Face2Face method.</center>
</p>

<p style="text-align: center">
    <img src="/uploads/upload_727953498e8fe33a6111483bfdbf6b10.png" alt>
    <center>Manipulated video using the FaceSwap method.</center>
</p>

### Testing:

- [Real and Fake Face Detection dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download)
    - Images where a part of the face has been altered with photoshop
    - Includes 4 subjectively categorized levels of difficulty for photoshop quality
    - 1081 real images that we use for training, 960 fake images for testing.
    - Images have been labeled to show which part of the face has been altered with

<p style="text-align: center">
    <img src="/uploads/upload_5f585d83aa6495cd61c99ecf0ca5b0db.jpeg" alt>
    <center>Figure 2. Example of how the images are labelled with the use of filenames.</center>
</p>

<p style="text-align: center">
    <img src="/uploads/upload_d17546e53a6537030a6993b5a538b9f2.png" alt>
    <center>Six examples of real images.</center>
</p>

<p style="text-align: center">
    <img src="/uploads/upload_c00eab921b64ae158dbe01ccb0a258c5.png" alt>
    <center>Six examples of easy, medium and hard photoshopped images, two per category.</center>
</p>


- [CelebA-DF](https://github.com/bomb2peng/DFGC_starterkit/tree/master/DFGC-21%20dataset)
    - Face images of celebrities collected from videos
    - Includes both real and fake images, where the fake images have been created using a variety of faceswap methods. Additionally adversarial noise has been added to morphed images to make detection harder
    - The dataset is not public, but can be accessed by filling in [form](https://docs.google.com/forms/d/e/1FAIpQLSdlHKqsvkpGtbm37KJdkaswWL-llOSqqZPaa8F5yJ08-koX2Q/viewform?usp=sf_link)


## Results

### For reproduction

### For extension

## Discussion & possible further improvements

## How we improved the current state

While replicating the original results, we noticed that the implementation was unreasonably slow, and using different GPU models yielded no significant speedups. We profiled the code using [scalene](https://github.com/plasma-umass/scalene) and realized the data loader had two major issues: the selected MRI/CT slices were normalized using the full volume statistics, and whole volumes were loaded into memory, even though only 8 slices per sample were used. To solve the first issue, we moved the normalization code to the preprocessing notebook, and in order to speed up the data loading, we converted the samples to the [HDF5](http://hdfgroup.org/) data format, which allowed us to only read a subsection of the data instead of loading the whole $256 \times 160 \times 160$ array. These two modifications reduced the training time for the VQ-VAE and the AR model to ~2.5 hours per model instead of 20+ hours using an NVIDIA A100 graphics card.

One downside of HDF5 files is that arrays can only be read in a strictly increasing order, and no duplicate reads are allowed. The original implementation selected the train slices using `random.choice`, which samples with replacement. We instead used `random.sample`. Thus for a given batch, no duplicate slices were selected. While this did not yield any visible improvements in the metrics (limited by the small validation set), this should reduce the model's overfitting.

Finally, the evaluation code for the pixel-wise anomaly scores ran a forward pass `model.forward_latent(samples, cond_repeat)[:, :, r, c]` for each unordered pair $(r, c)$ of latent codes. We updated the code to run a single forward pass per sample, reducing the evaluation time from 15 minutes to a few seconds.

## Individual contributions


## Bibliography
 
Chen, X., Mishra, N., Rohaninejad, M., & Abbeel, P. (2018, July). Pixelsnail: An improved autoregressive generative model. In International Conference on Machine Learning (pp. 864-872). PMLR.

Chen, X., You, S., Tezcan, K. C., & Konukoglu, E. (2020). Unsupervised lesion detection via image restoration with a normative prior. Medical image analysis, 64, 101713.
 
Kortli, Y., Jridi, M., Al Falou, A., and Atri, M. Face recognition systems: A survey. Sensors 20, 2 (2020), 342.
 
Marimont, S. N., & Tarroni, G. (2021, April). Anomaly detection through latent space restoration using vector quantized variational autoencoders. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI) (pp. 1764-1767). IEEE.

Tolba, A. S., El-Baz, A. H., & El-Harby, A. A. (2006). Face recognition: A literature review. International Journal of Signal Processing, 2(2), 88-103.

Parmar, R., Kuribayashi, M., Takiwaki, H., & Raval, M. S. (2022, July). On fooling facial recognition systems using adversarial patches. In 2022 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

Xue, J., Yang, Y., & Jing, D. (2019, August). Deceiving face recognition neural network with samples generated by deepfool. In Journal of Physics: Conference Series (Vol. 1302, No. 2, p. 022059). IOP Publishing.
 
Zimmerer, D., Full, P. M., Isensee, F., Jäger, P., Adler, T., Petersen, J., ... & Maier-Hein, K. (2022). MOOD 2020: A public Benchmark for Out-of-Distribution Detection and Localization on medical Images. IEEE Transactions on Medical Imaging, 41(10), 2728-2738.
