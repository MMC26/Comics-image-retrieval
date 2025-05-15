# Qualitative Test for Text-Image Matching Model

This document explains a Python script designed to qualitatively evaluate a text-image matching model. The script visualizes the model's ability to select the most relevant image given a textual input, using a small set of test examples.

## Workings

In this scripted a qualitative version of test is implemented due to the original dataset beloging to a competition and thus not providing the graound truth for this part. All functions have the possibility to use cuda and the in charge for the main pipeline can iterate over batches from a loader. The loader doesn't reuire the index label as explianed so just the text and the images sets are embedd using the trained model. Then similarity score is used to compare the embeddings we thought this is a representative metric to score the best fitting image. This is fundamentally a matrix or vector multiplication between embeddings so those with higher common features will be scoring highly, which is consistent with the model idea: correlating features in a shared embedding space. And so we select the highest scoring image to be the one most representative of the text. All the images show in a plot with the one being label as selected to interpret the results in aforemention qualitative manner. 

## Code 

The conde consists on three definitons of functions: 

*preprocess_image* : A path, tensor or open Img object can be provided to the function and will handle any of the formats to open the Image with PIL and convert it to RGB

*demo_test* : Function to be fed a model istance not on the device, a cheptoint path to load form a checkpoint.pt file, a list of image sets and texts. This function purpose is to use for small demonstrations of already trained models, diferent to the function used in main does not iterate on batches form a dataloader, ingeneered for small representations instead.

*qualitative_test*: Function used in the pipeline of the main training program, uses gpu to process batches. It outputs the first n desired samples.

## Example usage

Instaciate a model class and select the path for the checkpoint file. Create a custom list of text and images from local repository as an example and show the results via *demo_test* then results can be visulized for clarification and insight on the scorings

model = YourModelClass(...)
checkpoint_path = "path/to/your_checkpoint.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model_checkpoint(model, checkpoint_path, device)

samples = [
    {
        "text": "A cat sitting on a sofa",
        "image_set": ["cat1.jpg", "dog1.jpg", "cat2.jpg"]
    },
    {
        "text": "A red sports car",
        "image_set": [PILImage.open("car1.png"), PILImage.open("car2.png")]
    },
]

results = qualitative_demo(model, samples, device=device, num_examples=5)
---

