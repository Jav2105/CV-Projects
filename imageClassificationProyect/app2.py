import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def get_model(num_classes=23):
    #resnet 18
    model = models.resnet18(weights=None) 
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# model structure
model = get_model(num_classes=23).to(device)


model_path = "best_tree_model.pth" 

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: '{model_path}' not found.")
    exit()
except RuntimeError as e:
    print(f"Error loading weights: {e}")
    exit()

model.eval()

#class_names.json
class_names = [
    "Acer palmatum", "Cedrus deodara", "Celtis sinensis", 
    "Cinnamomum camphora (Linn) Presl", "Elaeocarpus decipiens", 
    "Flowering cherry", "Ginkgo biloba", "Koelreuteria paniculata", 
    "Lagerstroemia indica", "Liquidambar formosana", 
    "Liriodendron chinense", "Magnolia grandiflora L", 
    "Magnolia liliflora Desr", "Michelia chapensis", 
    "Osmanthus fragrans", "Photinia serratifolia", 
    "Platanus", "Prunus cerasifera f. atropurpurea", 
    "Salix babylonica", "Sapindus saponaria", 
    "Styphnolobium japonicum", "Triadica sebifera", 
    "Zelkova serrata"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_tree(image):
    if image is None:
        return None
    
    try:
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Create results dictionary
        results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        return results
    except Exception as e:
        return {f"Error: {str(e)}": 0.0}


if __name__ == "__main__":
    interface = gr.Interface(
        fn=predict_tree,
        inputs=gr.Image(type="pil"), 
        outputs=gr.Label(num_top_classes=3),
        title="🌿 Urban Tree Classifier (ResNet18)",
        description="Upload a photo of a street tree to identify its species.",
    )
    
    interface.launch()