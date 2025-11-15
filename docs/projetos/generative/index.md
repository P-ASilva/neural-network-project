# Text/Image to Video Model

## 1. Core Workflow and Components

This workflow uses a *latent video diffusion pipeline* to convert text prompts into a fully generated video. The model interprets the prompt, generates video frames inside latent space, and finally decodes them into real pixel images. The pipeline relies on four key components: the **Diffusion Model**, **CLIP**, **VAE**, and **Model Sampling**.

---

### 1.1. Diffusion Model

The diffusion model (WAN video model) is the engine responsible for creating the actual visual content and motion of the video. It operates fully in *latent space* rather than pixel space, which allows high-quality video synthesis with manageable compute.

The model begins from random latent noise and progressively refines it over several steps, shaping it into temporally consistent video frames. During refinement, it incorporates guidance from the prompt embeddings (positive and negative), enforcing visual style, subject appearance, lighting, scene layout, motion, and cinematic behavior.

Key responsibilities:

- Generates latent video frames from noise
- Maintains temporal and spatial consistency across frames
- Adapts to the prompt’s style, mood, and subject description

---

### 1.2. CLIP

CLIP serves as a text encoder, converting prompts into vector representations (embeddings) that the diffusion model can understand. It does not generate images or video by itself. Instead, it communicates the meaning of the prompt.

Two embeddings are used:

- **Positive Prompt Embedding**: pushes the model toward desired attributes, objects, and styles
- **Negative Prompt Embedding**: discourages unwanted artifacts such as blur, noise, distortions, or style deviations

CLIP determines what the video should or should not look like, and the diffusion model uses this information during the generative process.

---

### 1.3. VAE

The Variational Auto-Encoder (VAE) handles the conversion between latent space and pixel space. During video generation, the diffusion model outputs a sequence of latent frames rather than visible images. The VAE takes these latent frames and decodes them into real RGB visuals, acting as the final rendering stage of the pipeline.

Responsibilities of the VAE decoder:

- Converts latent video representations into full-resolution frames
- Preserves textures, sharpness, lighting, and color dynamics
- Adds visual detail that cannot be represented directly inside the diffusion model

Without the VAE, video frames would remain encoded numerical structures instead of displayable images.

---

### 1.4. Model Sampling

Model sampling determines how latent noise is iteratively refined into a coherent video. Sampling settings define the trajectory of denoising across time, impacting stability, detail, and style.

The sampling module controls parameters such as:

- Number of diffusion steps
- Type of sampler (e.g., uni_pc)
- Scheduler behavior
- Denoising strength and balancing between noise removal and prompt adherence
- Random seed and generation reproducibility

Sampling = the route the model takes from noise to final video. Different samplers and step counts influence cinematic smoothness, character consistency, and level of detail.


### 1.5. Model Details

The workflow is built on three core pretrained components: the diffusion model, the text encoder, and the VAE. Together, they define both the expressiveness of the video generation and the computational cost.

- **Diffusion Model** (Wan2.1-VACE-14B-Q3)  
  A large 14-billion-parameter transformer-based video diffusion model trained for text-to-video and image-to-video tasks.  
  The *Q3* quantized version allows the model to run with significantly reduced VRAM usage while maintaining high visual fidelity and temporal consistency.  
  Official files and weights used in this project can be found here:  
  **https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/tree/main**

- **Text Encoder** (UMT5-XXL FP8)  
  A very large T5-style language model that converts prompts into dense embeddings.  
  Its strong semantic understanding enables detailed, narrative, and stylistic control over the generated video.

- **VAE** (Wan-2.1 VAE)  
  A spatio-temporal Variational Auto-Encoder responsible for decoding latent video frames into full-resolution RGB outputs.  
  It preserves fine texture, sharpness, lighting, and color dynamics across frames, ensuring temporal coherence.

In combination, these models form a high-capacity video generation system capable of producing coherent motion, consistent subjects, and cinematic rendering while remaining computationally feasible through mixed quantization.


### 1.6. Finished workflow and results
This is a screenshot of the complete workflow:
![text to video workflow](t2v_workflow.png)

This is the postive prompt used:
```text
A sword embedded in a pile of burning wood, the fire moving naturally, glowing embers and smoke rising in a dark background, volumetric lighting, detailed textures
```
This is the negative prompt used:
```text
bad quality, blurry, messy, chaotic
```

This is the video generated by those prompts:
<video controls width="100%" style="max-width: 800px;">
  <source src="bonfire_vid.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>