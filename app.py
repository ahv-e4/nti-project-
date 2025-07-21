import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import streamlit as st

# 🧠 تحميل النموذج
class SkinCancerModel(nn.Module):
    def __init__(self):
        super(SkinCancerModel, self).__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)

# 🧠 تحميل الحالة المدربة
model = SkinCancerModel()

model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))


model.eval()

# 🌀 التحويلات المطلوبة للصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🎨 واجهة Streamlit
st.title("🔬 Skin Cancer Detection")
st.write("ارفع صورة لـ skin lesion وهقولك هل فيها سرطان ولا لأ")

uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)

    # تجهيز الصورة
    input_image = transform(image).unsqueeze(0)

    # توقع
    with torch.no_grad():
        output = model(input_image)
        prediction = torch.argmax(output, dim=1).item()

    labels = ["سليم", "سرطان جلد"]
    st.write(f"💡 النتيجة: **{labels[prediction]}**")
