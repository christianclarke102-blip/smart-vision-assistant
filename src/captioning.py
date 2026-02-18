from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def caption(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.decode(out[0], skip_special_tokens=True)
