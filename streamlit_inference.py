import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np

transforms_test = transforms.Compose([
    transforms.CenterCrop((1500, 800)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image =Image.open(image_name).convert('RGB')
    image = transforms_test(image)
#     image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image #.to(device)  #assumes that you're using GPU

## 데이타 체크
import torchvision
#import matplotlib.pyplot as plt
#def imshow(inp, title=None):
#    """Imshow for Tensor."""
#    inp = inp.numpy().transpose((1, 2, 0))
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    inp = std * inp + mean
#    inp = np.clip(inp, 0, 1)
#    plt.imshow(inp)
#    if title is not None:
#        plt.title(title)
#    plt.pause(0.001)  # pause a bit so that plots are updated

def onehot_label(row):
    p = re.compile('[A-Z]+')
    multi_label = row.split('/')[3].split('_')[1]
    multi_label = p.findall(multi_label)
    multi_label_list = torch.LongTensor([ord(alpha)-65 for alpha in multi_label[0]])
    y_onehot = torch.nn.functional.one_hot(multi_label_list, num_classes=10)
    y_onehot = np.array(y_onehot.sum(dim=0).float())
    return y_onehot

def onehot2abc(row):
    if type(row) == str:
        row= onehot_label(row)
    argmax_= np.where(row==1)
    return list(map(chr, [x+65 for x in argmax_[0].tolist()] ))

@st.cache
def load_model():
    from efficientnet_pytorch import EfficientNet

    model_name = 'efficientnet-b3'
    save_model_name = 'eff-b3'
    num_classes = 10
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

    weights_path = 'best_model_eff3_v=t.pt'
    state_dict = torch.load(weights_path, map_location='cpu') # , map_location=device)  # load weight
    #model.load(state_dict)
    model.load_state_dict(state_dict, strict=False)  # insert weight to model structure
    #model = model.to(device)
    return model


def inference(image_filename):
    image = image_loader(image_filename)
    #imshow(image.cpu().squeeze())
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs = image #.to(device)
        outputs = model(inputs)
        preds = [1 if x > 0.5 else 0 for x in
                 outputs.squeeze().tolist()]  # the class with the highest energy is what we choose as prediction
    return onehot2abc(np.array(preds))






st.title('블록 패턴 추출')
model = load_model()

st.markdown('↓↓↓ 카메라로 직접 블록구조를 촬영하실 수 있습니다.')

if st.button("📸 Camera"):
    picture = st.camera_input("Take a picture")
    st.image(picture)
    if picture:
        uploaded_file = picture
        #image = Image.open(uploaded_file)
        st.image(uploaded_file, caption='Input Image', use_column_width=True)
        # st.write(os.listdir())
        answer = inference(uploaded_file)
        st.write(f"패턴: {answer}")

else:
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # src_image = load_image(uploaded_file)
        
        st.image(uploaded_file, caption='Input Image', use_column_width=True)
        # st.write(os.listdir())


        answer = inference(uploaded_file)

        st.write(f"패턴: {answer}")







