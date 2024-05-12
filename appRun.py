from openvino.inference_engine import IECore
from openvino.runtime import Core
import cv2
from fastai.vision.all import *
import pathlib
import PIL
from PIL import Image
from matplotlib import cm
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn=load_learner("convnext_tiny_small_size_all_dataset.pkl")



onnx_file_name = f"E:\RunOpenVino\Models\ConvexTiny-AllData-Optimized-52ms\Convnext-all.onnx"
class_labels_file_name = f"E:\RunOpenVino\Models\ConvexTiny-AllData-Optimized-52ms\Convnext-all-classes.json"
ir_path = f"E:\RunOpenVino\Models\ConvexTiny-AllData-Optimized-52ms\Convnext-all.xml"
ir_path


# openvino
# # Load the network in Inference Engine
# ie = Core()
# model_ir = ie.read_model(model=ir_path)
# model_ir.reshape(torch.Size([1, 3, 124, 124]))
# compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

# # Get input and output layers
# input_layer_ir = next(iter(compiled_model_ir.inputs))
# output_layer_ir = next(iter(compiled_model_ir.outputs))


# ##onnx
# ie = Core()
# model_onnx = ie.read_model(model=onnx_file_name)
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

# input_layer_onnx = next(iter(compiled_model_onnx.inputs))
# output_layer_onnx = next(iter(compiled_model_onnx.outputs))




cap = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

preds=["3asfor", "Black-grass", "Bots", "Carolin", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Darasina", "Darasina Zora", "Darasina limon", "FAKE", "Fat Hen", "Iglonima", "Lamon atalya", "Limon", "Loose Silky-bent", "Maize", "Mango keet", "Petra", "Scentless Mayweed", "Sheflyra", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet", "bashamela", "borto2an", "darasina sandroza", "dre2a 3alya", "gwafa", "mango zebdya", "salomy eshta", "yosfe", "zaton"]
while True:
    ret, frame = cap.read()
    
    
    #openvino 
    # Run inference on the input image
    # test_img = Image.fromarray(np.uint8(frame)).convert('RGB')
    # resized_img = test_img.resize([124,124])
    
    # img_tensor = tensor(resized_img).permute(2, 0, 1)
    # # res = compiled_model_onnx(inputs=[img_tensor.unsqueeze(dim=0)])[output_layer_onnx]
    # res= compiled_model_ir([img_tensor.unsqueeze(dim=0)])[output_layer_ir]
    # res=preds[np.argmax(res)]
    


    # ##onnix
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # test_img = Image.fromarray(np.uint8(frame)).convert('RGB')
    # resized_img = test_img.resize([224,224])
    # img_tensor = tensor(resized_img).permute(2, 0, 1)
    # scaled_tensor = img_tensor.float().div_(255)
    # mean_tensor = tensor(mean).view(1,1,-1).permute(2, 0, 1)
    # std_tensor = tensor(std).view(1,1,-1).permute(2, 0, 1)
    # normalized_tensor = (scaled_tensor - mean_tensor) / std_tensor
    # batched_tensor = normalized_tensor.unsqueeze(dim=0)
    # normalized_input_image = batched_tensor.cpu().detach().numpy()
    # res = compiled_model_onnx(inputs=[normalized_input_image])[output_layer_onnx]




    ##learn.predict
    res=learn.predict(cv2.resize(frame,(124,124)))[0]

    if  res !=None:
        cv2.putText(frame,res,(20,30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
  
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, "Fps="+fps, (7, 70), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('webcam feed' , frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
