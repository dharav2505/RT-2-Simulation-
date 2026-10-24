import torch
import clip
from PIL import Image

class CLIPPerception:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")

    def match_command_to_object(self, image, command, object_labels):
        image_inputs = [self.preprocess(image).unsqueeze(0)]
        text_inputs = torch.cat([clip.tokenize(f"{command} {obj}") for obj in object_labels])
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs[0])
            text_features = self.model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()
        return object_labels[best_idx]
