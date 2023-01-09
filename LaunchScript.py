import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionImageVariationPipeline
def main(
        input_im,
        scale=3.0,
        n_samples=5,
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
    """

article = \
    """
    """

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
)
pipe = pipe.to(device)
inputs = [
    gr.Image(),
    gr.Slider(0, 30, value=3, step=1, label="Guidance Scale"),
    gr.Slider(1, 5, value=1, step=1, label="Number of Images"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, label="Seed", precision=0)
]
output = gr.Gallery(label="Generated Variations")
output.style(grid=2)
examples = [
    ["D:/testimage.jpg", 3, 1, 25, 0],
]

demo = gr.Interface(
    fn=main,
    title="smart plagiarism",
    description=description,
    article=article,
    inputs=inputs,
    outputs=output,
    examples=examples,
)
demo.launch()