"""
    Use the `transformers` library to instantiate and get inference scores for different variants of the `clip-vit` family of zero-shot-classification models
@ref https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/
"""
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

# Model variant
model_variant = "openai/clip-vit-base-patch32"

# Load model, tokeniser, and processor
model = CLIPModel.from_pretrained(model_variant)
processor = AutoProcessor.from_pretrained(model_variant)
tokenizer = AutoTokenizer.from_pretrained(model_variant)

# Get image/text similarity softmax output
def image_text_relevance(image_path:str, text_choices:list[str]):
    global processor, model
    img = Image.open(image_path)
    inputs = processor(
        text = text_choices,
        images = img,
        return_tensors = "pt",
        padding = True
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs

# Get image feature vector
def image_features(image_path:str):
    global model, processor
    img = Image.open(image_path)
    inputs = processor(
        images = image,
        return_tensors = "pt",
        padding = True
    )
    image_features = model.get_image_features(**inputs) # image features
    return image_features
