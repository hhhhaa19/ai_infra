import os
import urllib.request
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import onnx

def main():
    os.makedirs("data", exist_ok=True)
    img_path = "data/cat.jpg"
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Downloading sample image from {url} ...")
    urllib.request.urlretrieve(url, img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    print("==========================================")
    print(f"Input Tensor Shape: {input_batch.shape}")
    print(f"Input Dtype:        {input_batch.dtype}")
    
    print("Loading pretrained ResNet-50 ...")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    
    onnx_path = "data/resnet50.onnx"
    torch.onnx.export(
        model,
        (input_batch,),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX saved: {onnx_path}")
    print("ONNX structure check passed.")
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    print("PyTorch Prediction:")
    categories = weights.meta["categories"]
    for i in range(top3_prob.size(0)):
        print(f"  Top {i+1}: {categories[top3_catid[i]]} ({top3_prob[i].item()*100:.2f}%)")
    print("==========================================")
    
    input_np = input_batch.contiguous().cpu().numpy().astype(np.float32)
    output_np = output.contiguous().cpu().numpy().astype(np.float32)
    
    input_bin_path = "data/input.bin"
    output_bin_path = "data/output_ref.bin"
    
    input_np.tofile(input_bin_path)
    output_np.tofile(output_bin_path)
    
    print(f"Successfully saved: {input_bin_path} ({input_np.nbytes} bytes)")
    print(f"Successfully saved: {output_bin_path} ({output_np.nbytes} bytes)")
    
    
    print("\n[Sanity Check - First 5 Floats]")
    print("Input Tensor first 5: ", input_np.flatten()[:5])
    print("Output Tensor first 5:", output_np.flatten()[:5])
    print("==========================================")

if __name__ == "__main__":
    main()