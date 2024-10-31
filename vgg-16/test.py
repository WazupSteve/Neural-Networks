import torch
from torchvision import transforms
from PIL import Image
from main import VGG16 

def predict_image(image_path, model_path, num_classes=100):
    #load model
    model = VGG16(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    #transform image
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])
    
    #load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    #get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return prediction.item(), confidence.item()

if __name__ == "__main__":

    IMAGE_PATH = "./test.jpeg"
    MODEL_PATH = "vgg-16/model_checkpoint_epoch_20.pth"
    
    prediction, confidence = predict_image(IMAGE_PATH, MODEL_PATH)
    print(f"Predicted class: {prediction}")
    print(f"Confidence: {confidence:.2%}")