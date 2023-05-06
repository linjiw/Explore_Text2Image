# Project Report: Test 2 Image Generating Task 🎨🖼️

![Imagen Architechture](./data/Imagen_model_structure.png "Imagen Model")
<figure>
  <img src="https://www.example.com/image.jpg" alt="Alt text" title="Optional title">
  <figcaption>Here is the caption for the image.</figcaption>
</figure>


## Introduction

In this project, we aim to implement an **Image Generation** model based on the [**MinImagen**][1](https://github.com/AssemblyAI-Examples/MinImagen) architecture. MinImagen is a text-to-image generation model that is efficient and generates high-quality images from textual descriptions. Our goal is to experiment with this model and share our findings and experiences.

## Implementation

We began by implementing the **MinImagen** model using the resources provided by **AssemblyAI** [2]. The key component of this implementation is the **Diffusion** part, which has the following characteristics compared with the **GigaGAN** in our training environment.:

- Training is fast ⚡
- Sample image generation is slow 🐢
- Convergence is fast 🏃
- Generated results are more meaningful 🎯

<!-- Another important feature of our implementation is the **classifier-free guidance** for text conditioning. This means that we don't need to rely on a separate classifier to guide the image generation process. -->

### Dataset

We used the **HuggingFace Pokémon caption dataset** 🐾 as the basis for our image generation task. We edited the dataloader and dataset functions to make them compatible with our implementation.

### Experimentation

During the experimentation phase, we encountered some challenges:

1. **Model Size**: The original model was too large to fit into a 12GB GPU for training, even with a batch size of 1.
2. **Logging**: We used **Weights & Biases (wandb)** for logging our training process.
3. **Super Resolution**: Due to the model's large size, we had to remove the super resolution layer for training.

Despite these challenges, our model was able to converge quickly, taking only 300 epochs with a batch size of 2 and a time step of 1000 to generate meaningful images. 🌟

## Future Work 💡

We plan to add a **super resolution layer** in the future to further improve our image generation capabilities.

## Conclusion

Our project on implementing and experimenting with the MinImagen architecture for text-to-image generation has been successful. We were able to generate meaningful images from textual descriptions, overcoming challenges related to model size and training resources. We hope that our experience and findings can help others working on similar projects. 😃

[1]: MinImagen: Build Your Own Imagen Text-to-Image Model (https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/)
[2]: AssemblyAI-Examples/MinImagen GitHub Repository (https://github.com/AssemblyAI-Examples/MinImagen?ref=assemblyai.com)
