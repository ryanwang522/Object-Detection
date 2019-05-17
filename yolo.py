import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
import time
import models

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print("Device used: ", device)

OBJECT_CLASSES = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "small-vehicle": 9,
    "large-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15
}

class Aerial(Dataset):
    def __init__(self, root, grid_size=7, transform=None, test_mode=False):
        """ Intialize the Aerial dataset """
        self.root = root
        self.transform = transform
        self.grid_size = grid_size
        print('Grid size: ', self.grid_size)
        # read filenames
        self.filenames = glob.glob(os.path.join(root, '*.jpg'))
        self.len = len(self.filenames)
        self.label_root = "/".join(self.root.split('/')[:-2])
        self.test = test_mode
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = Image.open(self.filenames[index])
        
        label = np.zeros(shape=(self.grid_size, self.grid_size, 26))
        
        # get label's filename
        file_idx = self.filenames[index].split('/')[-1][:-4] 
        label_fn = file_idx + '.txt'
        
        def scale(coord):
            """ scale the coordinates of x, y wrt the grid it belongs to """
            new_coords = math.modf(coord / 512 * self.grid_size) 
            return new_coords[0], int(new_coords[1])
        
        if self.test == False:
            # read label's file
            with open(os.path.join(self.label_root, 'labelTxt_hbb', label_fn)) as label_f:
                for line in label_f:
                    # discard difficulty
                    line = line.split()[:-1]
                    # turn class name into index 
                    line[-1] = OBJECT_CLASSES[line[-1]]
                    class_vec = [0] * 16
                    class_vec[line[-1]] = 1.0
            
                    line = np.asarray(list(map(float, line)))

                    xmin, ymin = line[0], line[1]
                    xmax, ymax = line[4], line[5]
                    x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
                    w, h = xmax - xmin, ymax - ymin 
                                    
                    # turn x, y into the coordinates wrt its grid
                    cell_x, lb_x = scale(x)
                    cell_y, lb_y = scale(y)

                    # turn w, h into [0:1]
                    w, h = w / 512, h / 512
                    label[lb_x][lb_y][:] = np.asarray([cell_x, cell_y, w, h, 1., cell_x, cell_y, w, h, 1.] + class_vec) 

        if self.transform is not None:
            image = self.transform(image)

        return image, label.astype(np.float32), file_idx

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def IoU(bboxes, target, grid_size):
    xt, yt, wt, ht = target[0:4] # ground truth
    wt = wt * grid_size
    ht = ht * grid_size
    ious = []
    for x, y, w, h in bboxes:
        if (abs(x - xt) < (w + wt) / 2.0) and (abs(y - yt) < (h + ht) / 2.0):
            
            lu_x_inter = max((x - (w / 2.0)), (xt - (wt / 2.0)))
            lu_y_inter = min((y + (h / 2.0)), (yt + (ht / 2.0)))
    
            rd_x_inter = min((x + (w / 2.0)), (xt + (wt / 2.0)))
            rd_y_inter = max((y - (h / 2.0)), (yt - (ht / 2.0)))
    
            inter_w = abs(rd_x_inter - lu_x_inter)
            inter_h = abs(lu_y_inter - rd_y_inter)
    
            inter_square = inter_w * inter_h
            union_square = (w * h) + (w * h) - inter_square
    
            IOU = inter_square / union_square * 1.0
            ious.append(IOU)
        else:
            ious.append(0.0)
    
    # ious is a list of tensors
    return ious

def loss_fn(predict, target, grid_size=7):
    batch_size, _, _, _ = predict.shape
    lambda_coord, lambda_noobj = 5, 0.5

    obj_mask = target[:, :, :, 4] == 1.0
    noobj_mask = target[:, :, :, 4] == 0.0

    obj_target = target[obj_mask] # (x, 26)
    obj_predict = predict[obj_mask]
    noobj_target = target[noobj_mask]
    noobj_predict = predict[noobj_mask]

    # class prediction loss
    class_loss = F.mse_loss(obj_predict[:, 10:], obj_target[:, 10:], reduction='sum')

    # find the bbox with bigger IoU and calculate cell's bbox prediction loss       
    bbox_mask = torch.zeros((obj_target.shape), dtype=torch.int8)
    iou_table = torch.zeros((obj_target.shape[0], 2)).to(device)
    for i, (pred_vec, targ_vec) in enumerate(zip(obj_predict, obj_target)):
        bboxes = []
        bboxes.append(tuple(pred_vec[0:2]) + tuple(pred_vec[2:4] * grid_size)) # x1, y1, w1, h1
        bboxes.append(tuple(pred_vec[5:7]) + tuple(pred_vec[7:9] * grid_size)) # x2, y2, w2, h2

        # calculate IoU and get the bbox index with different IoU
        iou = IoU(bboxes, targ_vec[0:4], grid_size)
        iou_table[i][0] = iou[0]
        iou_table[i][1] = iou[1]
        bbox_idx = torch.argmax(iou_table[i])
        for j in range(10):
            bbox_mask[i][j] = 1 if j // 5 == bbox_idx else -1
        
    
    bigIoU_mask = bbox_mask[:, :] == 1
    smallIoU_mask = bbox_mask[:, :] == -1

    # bbox vectors with bigger IoU and smaller IoU
    obj_pred_big = obj_predict[bigIoU_mask].contiguous().view(bbox_mask.shape[0], 5)
    obj_pred_small = obj_predict[smallIoU_mask].contiguous().view(bbox_mask.shape[0], 5)

    # coordinate loss
    coord_loss = lambda_coord * F.mse_loss(
        obj_pred_big[:, :2], obj_target[:, :2], reduction='sum')
    coord_loss += lambda_coord * F.mse_loss(
        torch.sqrt(obj_pred_big[:, 2:4]), torch.sqrt(obj_target[:, 2:4]), reduction='sum')
    
    bigIoU = torch.max(iou_table, 1)[0]
    smallIoU = torch.min(iou_table, 1)[0]
    confid_loss = F.mse_loss(obj_pred_big[:, 4] * bigIoU[:], obj_target[:, 4] * bigIoU[:], reduction='sum')
    confid_loss += F.mse_loss(obj_pred_small[:, 4] * smallIoU[:], torch.zeros((obj_pred_small.shape[0])).to(device), reduction='sum' )
    
    # decrease the cells don't have any gt detections
    noobj_confid_loss = lambda_noobj * F.mse_loss(noobj_predict[:, 4], noobj_target[:, 4], reduction='sum')
    noobj_confid_loss += lambda_noobj * F.mse_loss(noobj_predict[:, 9], noobj_target[:, 9], reduction='sum')

    return (coord_loss + confid_loss + noobj_confid_loss + class_loss) / batch_size


def save_ckpt(ckpt_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, ckpt_path)
    print('model saved to %s\n' % ckpt_path)

def train(model, epoch, log_interval=200, start_ep=0, grid_size=7):
    trainset = Aerial(root='./hw2_train_val/train15000/images/', transform=transforms.ToTensor(), grid_size=grid_size)
    print('# images in trainset: ', len(trainset))

    trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = loss_fn
    model.train()

    #ckpt_epochs = np.arange(epoch)
    iteration = 0
    for ep in range(epoch):
        start_time = time.time()
        for batch_idx, (data, target, _) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predict = model(data)
            loss = criterion(predict, target, grid_size)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        print('Time: {:.4f}'.format((time.time() - start_time) / 60.0))
        valid_loss = validate(model, grid_size)
        start_time = time.time()
        save_ckpt('./ckpts/ckpt_ep{}_vloss-{:.4f}.pth'.format(ep+start_ep, valid_loss), model, optimizer)
        print('Time for saving model: {:.4f}'.format((time.time() - start_time) / 60.0))

    # save final model
    torch.save(model.state_dict(), './model/final_model.pkl')


def validate(model, grid_size):
    criterion = loss_fn
    model.eval()
    valid_loss = 0
    valset = Aerial(root='./hw2_train_val/val1500/images/', transform=transforms.ToTensor(), grid_size=grid_size)
    valset_loader = DataLoader(valset, batch_size=50, shuffle=False, num_workers=1)

    with torch.no_grad():
        for data, target, _ in valset_loader:
            data, target = data.to(device), target.to(device)
            predict = model(data)
            valid_loss += criterion(predict, target, grid_size=grid_size)

    valid_loss /= (len(valset_loader.dataset) / 50)
    print('\nValidation set - Avg loss: {:.4f}'.format(valid_loss))    
    return valid_loss


def gen_pred_file(model, pred_path='./hw2_train_val/val1500/predTxt_hbb/', anno_path='./hw2_train_val/val1500/images/', grid_size=7):
    model.eval()
    valset = Aerial(root=anno_path, transform=transforms.ToTensor(), grid_size=grid_size, test_mode=True)
    valset_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        for (data, _, file_idx) in valset_loader:
            
            file_idx = file_idx[0]
            data = data.to(device)
            predict = model(data)[0] # predict - 7x7x26    

            # turn the predict tensor in to bbox info
            # calculate class-specifc confidence score
            scores = np.zeros((grid_size, grid_size, 2))
            class_scores = torch.max(predict[:, :, 10:], dim=2, keepdim=True)[0] # 7x7x1 
            class_scores = torch.squeeze(class_scores)
            scores[:, :, 0] = (class_scores * predict[:, :, 4]).detach().cpu().numpy()
            scores[:, :, 1] = (class_scores * predict[:, :, 9]).detach().cpu().numpy()

            # threshold with 0.1
            scores[ scores < 0.17 ] = 0.

            # NMS
            pick = []
            threshold = 0.5
            while scores.any() > 0:
                # get the index of max score
                ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape) # e.g. (1, 2, 0)
                score = scores[ind]
                scores[ind] = 0

                # get the bbox coord of the score
                start, end = ind[2] * 5, 2 + ind[2] * 5 # 0~2 or 5~7
                pred_vec = predict[ind[0]][ind[1]]
                ind_vec = torch.tensor([ind[0], ind[1]], dtype=pred_vec.dtype).to(device)
                max_score_bbox = tuple(pred_vec[start:end] + ind_vec) + tuple(pred_vec[start+2:end+2]) # x, y, w, h
                pick.append((predict[ind[0], ind[1], :], ind, score)) # 紀錄是某個 grid 中第幾(ind[2])個 bbox and its score 

                # get the remain score
                sc_x, sc_y, sc_z = np.where(scores > 0)
                for m, n, q in zip(sc_x, sc_y, sc_z):
                    # m, n, q is index of scores
                    p_vec = predict[m, n, :]
                    start, end = q * 5, 2 + q * 5
                    ind_vec = torch.tensor([m, n], dtype=p_vec.dtype).to(device)
                    score_box = tuple(p_vec[start:end] + ind_vec) + tuple(p_vec[start+2:end+2] * grid_size) # x, y, w, h

                    # calculate IOU and performe the thresholding 
                    iou = IoU([score_box], max_score_bbox, grid_size)[0]
                    if iou > threshold:
                        scores[m, n, q] = 0

            # write prediction file
            with open(pred_path + file_idx + '.txt', 'w+') as out_f:
                for pred_vec, index, score in pick:
                    grid_x, grid_y, bbox_idx = index
                    start, end = bbox_idx * 5, 2 + bbox_idx * 5
                    x, y = (pred_vec[start:end]).detach().cpu().numpy()
                    x = (x + grid_x) * (512 / grid_size)
                    y = (y + grid_y) * (512 / grid_size)
                    w, h = (pred_vec[start+2:end+2] * 512).detach().cpu().numpy()

                    xmin, xmax = int(x - w / 2), int(x + w / 2)
                    ymin, ymax = int(y - h / 2), int(y + h / 2 )
                    coordinates = '{}.0 {}.0 {}.0 {}.0 {}.0 {}.0 {}.0 {}.0 '.format(
                        xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax
                    )

                    class_idx = torch.argmax(pred_vec[10:]).item()
                    class_name = [name for name, idx in OBJECT_CLASSES.items() if class_idx == idx][0]
                    score_str = ' {:2f}'.format(score)
                    out_f.write(coordinates + class_name + score_str + '\n')

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest='train', help="Train a model from scatch", action="store_true", default=False)
    parser.add_argument("--imp", dest='imp', help="Train an improved model from scatch", action="store_true", default=False)
    parser.add_argument("--continue", dest='ctn', help="Train a model from existing model", action="store_true", default=False)
    parser.add_argument("--eval", dest='eval', help="Generate predict files", action="store_true", default=False)
    parser.add_argument('--ckpt', action='store', dest='ckpt_path', default=None)
    parser.add_argument('-t', action='store', dest='test_dir', default='./hw2_train_val/val1500/images/')
    parser.add_argument('-p', action='store', dest='pred_dir', default='./hw2_train_val/val1500/predTxt_hbb/')
    parser.add_argument('-g', action='store', dest='grid_size', default=7)
    parser.add_argument('-ep', action='store', dest='epochs', default=30)
    args = parser.parse_args()

    if int(args.grid_size) == 8:
        print('Running improved model')
        model = models.Improved_model(pretrained=True).to(device)
    else:
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    
    if args.train:
        print('Training start')
        train(model, int(args.epochs), grid_size=int(args.grid_size))
    else:
        if args.ckpt_path == None:
            print("Need ckpt path!")
        elif args.eval:
            print('Test with: ', args.ckpt_path.split('/')[-1])
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            state = torch.load(args.ckpt_path)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            gen_pred_file(model, pred_path=args.pred_dir, anno_path=args.test_dir, grid_size=int(args.grid_size))
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            state = torch.load(args.ckpt_path)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            start_ep = int(args.ckpt_path.split('_')[2:])
            train(model, int(args.epochs), start_ep=start_ep+1)
