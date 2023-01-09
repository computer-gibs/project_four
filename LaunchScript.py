import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionImageVariationPipeline


def main(
        input_im,
        scale=3.0,
        n_samples=4,
        steps=25,
        seed=0,
):
    generator = torch.Generator(device=device).manual_seed(int(seed))

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(input_im).to(device)

    images_list = pipe(
        inp.tile(n_samples, 1, 1, 1),
        guidance_scale=scale,
        num_inference_steps=steps,
        generator=generator,
    )

    images = []
    for i, image in enumerate(images_list["images"]):
        if (images_list["nsfw_content_detected"][i]):
            safe_image = Image.open(r"unsafe.png")
            images.append(safe_image)
        else:
            images.append(image)
    return images


description = \
    """
    __Now using Image Variations v2!__
    Generate variations on an input image using a fine-tuned version of Stable Diffision.
    Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)
    This version has been ported to 🤗 Diffusers library, see more details on how to use this version in the [Lambda Diffusers repo](https://github.com/LambdaLabsML/lambda-diffusers).
    For the original training code see [this repo](https://github.com/justinpinkney/stable-diffusion).
    ![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)
    """

article = \
    """
    ## How does this work?
    The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
    the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
    This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
    This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.
    The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
    Training was done on 8xA100 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
    More details are on the [model card](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
    """

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
)
pipe = pipe.to(device)

inputs = [
    gr.Image(),
    gr.Slider(0, 25, value=3, step=1, label="Guidance scale"),
    gr.Slider(1, 4, value=1, step=1, label="Number images"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, label="Seed", precision=0)
]
output = gr.Gallery(label="Generated variations")
output.style(grid=2)

examples = [
    ["D:/testimage.jpg", 3, 1, 25, 0],
]

demo = gr.Interface(
    fn=main,
    title="Stable Diffusion Image Variations",
    description=description,
    article=article,
    inputs=inputs,
    outputs=output,
    examples=examples,
)
demo.launch()