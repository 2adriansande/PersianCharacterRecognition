import sys
import torch, torchvision, cv2, numpy, pandas, sklearn, streamlit

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("OpenCV      :", cv2.__version__)
print("CUDA ready  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

# Check that Persian text survives in a Python string
s = "\u067e\u0686\u0698\u06af"
print("Unicode test:", s, [hex(ord(c)) for c in s])