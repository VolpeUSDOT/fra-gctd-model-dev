import torch
import random
import numpy as np
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
from engine import train_one_epoch, evaluate
import utils
from PIL import Image
from fra_gctd_seg_dataset import gctd_dataset

# Basic settings
num_example_images = 10                         # Number of test images to annoate and save for manual inspection
num_classes = 2                                 # Number of classes in the dataset
num_epochs = 10                                 # Number of epochs to train the model

# file locations
dataset_basepath = '../../temp/gctd_seg'

# category names
CATEGORY_NAMES = [
    'GradeCrossing', 'RightOfWay'
]
LABEL_COLORS = [[0, 0, 255],[0, 255, 0],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180]]

def pil_to_cv(image):
    new_image = np.array(image)
    return new_image[:, :, ::-1].copy()

def colour_masks(image, color):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = color
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def parse_seg_prediction(pred, threshold):
    pred_score = list(pred['scores'].detach().cpu().numpy())
    print(pred_score)
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]

    masks = []

    if len(pred['masks']) > 1:
        masks = (pred['masks']>0.5).squeeze().detach().cpu().numpy()
    else:
        masks.append(pred['masks'][0][0].detach().cpu().numpy())
    
    pred_class = [CATEGORY_NAMES[i] for i in list(pred['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]

    labels = pred_class.copy()
    new_masks = []
    new_class = []
    new_boxes = []
    n = 0
    for label in pred_class:
        if label in CATEGORY_NAMES:
            new_masks.append(masks[n])
            new_boxes.append(pred_boxes[n])
            new_class.append(pred_class[n])
        n += 1
    return new_masks, new_boxes, new_class

def instance_segmentation_visualize(img, predictions, threshold=0.00001, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls = parse_seg_prediction(predictions, threshold)
    for i in range(len(masks)):
        rgb_mask = colour_masks(masks[i], LABEL_COLORS[i])
        rgb_mask = rgb_mask.transpose(2,0,1)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    return img

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset and defined transformations
dataset = gctd_dataset(dataset_basepath, get_transform(train=True))
dataset_test = gctd_dataset(dataset_basepath, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# save model
torch.save(model.state_dict(), 'gctd_grade-row.pt')

# test to make sure the model saved correctly
model.load_state_dict(torch.load('gctd_grade-row.pt'))
# put the model in evaluation mode
model.eval()
model.to(device)

# pick several random images from the test set for manual inspection
for x in range(num_example_images):

    img, _ = random.choice(dataset_test)

    # Run tests to see how our model performs
    with torch.no_grad():
        prediction = model([img.to(device)])
        img = (img.numpy() * 255).round().astype(np.uint8)
        final_image = instance_segmentation_visualize(img,prediction[0])
        final_image = final_image.transpose(1,2,0)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        example_image_filename='test' + str(x) + '.jpg'
        cv2.imwrite(example_image_filename, final_image)