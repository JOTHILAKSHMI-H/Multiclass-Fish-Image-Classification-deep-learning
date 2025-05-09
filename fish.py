import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# Class names
class_names = [
    "animal fish", 
    "animal fish bass",
    "fish sea_food black_sea_sprat", 
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet", 
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# Model architecture
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    num_classes = len(class_names)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load('mobilenetv2_fish_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction logic
def predict(image, model):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
    return class_names[predicted.item()], confidence

# Streamlit app
st.title("Multiclass Fish Image Classification")
uploaded_file = st.file_uploader('Upload an image of a fish', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=250)
    
    model = load_model()  # Load the pre-trained model
    label, confidence = predict(image, model)  # Get prediction and confidence
    
    st.write(f"Predicted: **{label}** with confidence **{confidence:.2f}**")
