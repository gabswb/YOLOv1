import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from model.yolo import Yolo
from model.loss import Loss
from data.data import VOCDetection

from tqdm.notebook import tqdm as tqdm2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime

# Configurations
root = '.'
run_name = 'vgg16'          # experiment name.
ckpt_root = 'checkpoints'   # from/to which directory to load/save checkpoints.
data_root = 'dataset'       # where the data exists.
pretrained_backbone_path = 'weights/vgg_features.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001          # learning rate
batch_size = 64     # batch_size
last_epoch = 0      # the last training epoch. (default: 0)
max_epoch = 50     # maximum epoch for the training.

num_boxes = 2       # the number of boxes for each grid in Yolo v1.
num_classes = 20    # the number of classes in Pascal VOC Detection.
grid_size = 7       # 3x224x224 image is reduced to (5*num_boxes+num_classes)x7x7.
lambda_coord = 7    # weight for coordinate regression loss.
lambda_noobj = 0.5  # weight for no-objectness confidence loss.

ckpt_dir = os.path.join(root, ckpt_root)


# for best ckpt
best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')
best_loss =  torch.load(best_ckpt_path)['val_loss'] if os.path.isfile(best_ckpt_path) else 5.0


# run the following script only at the initialization of the machine
ckpt_dir_ = ckpt_dir.replace(' ', '\\ ')
#!ln -sr {ckpt_dir_} ./

train_dset = VOCDetection(root=data_root, split='train')
train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

test_dset = VOCDetection(root=data_root, split='test')
test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)


def train(): 
    # for tensorboard record
    log_dir = Path(root) / 'logs' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    log_dir_ = str(log_dir).replace(" ", "\\ ")
    #%load_ext tensorboard
    #%tensorboard --logdir {log_dir_} --port 6007

    # for best ckpt
    best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')
    best_loss =  torch.load(best_ckpt_path)['val_loss'] if os.path.isfile(best_ckpt_path) else 5.0
    print("start best_loss : {}".format(best_loss))

    # Training & Testing.
    model = model.to(device)
    save_names = [name for name, v in model.named_parameters() if v.requires_grad is True]
    global_step = 0
    for epoch in range(max_epoch):
        # Learning rate scheduling
        if epoch in [195, 210]:
            lr *= 0.25
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch < last_epoch:
            continue

        model.train()
        batch_cnt = 0
        train_loss = 0.
        train_loss_sub = torch.zeros(size=(5,), dtype=torch.float, device=device)
        print("[{:02}/{:02}] epoch ==================== ".format(epoch + 1, max_epoch))

        for x, y in tqdm2(train_dloader):
            x = x.to(device) 
            y = y.to(device)
            
            prediction = model.forward(x)

            loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = compute_loss.forward(prediction, y)
            loss = lambda_coord * loss_xy + lambda_coord * loss_wh + loss_obj + lambda_noobj * loss_noobj + loss_class


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_loss_sub[0] += loss_xy.item()
            train_loss_sub[1] += loss_wh.item()
            train_loss_sub[2] += loss_obj.item()
            train_loss_sub[3] += loss_noobj.item()
            train_loss_sub[4] += loss_class.item()

            batch_cnt+=1

            # tensorboard record
            global_step += 1
            writer.add_scalar('train_loss/loss', train_loss, global_step)
            writer.add_scalar('train_loss/loss_xy', loss_xy.item(), global_step)
            writer.add_scalar('train_loss/loss_wh', loss_wh.item(), global_step)
            writer.add_scalar('train_loss/loss_obj', loss_obj.item(), global_step)
            writer.add_scalar('train_loss/loss_noobj', loss_noobj.item(), global_step)
            writer.add_scalar('train_loss/loss_class', loss_class.item(), global_step)

        train_loss /= batch_cnt
        train_loss_sub /= batch_cnt

        model.eval()
        batch_cnt = 0
        val_loss = 0.
        val_loss_sub = torch.zeros(size=(5,), dtype=torch.float, device=device)
        with torch.no_grad():
            for x, y in tqdm2(test_dloader):
                x = x.to(device) 
                y = y.to(device)

                prediction = model.forward(x)
                loss_xy, loss_wh, loss_obj, loss_noobj, loss_class = compute_loss.forward(prediction, y)
                loss = lambda_coord * loss_xy + lambda_coord * loss_wh + loss_obj + lambda_noobj * loss_noobj + loss_class
                
                val_loss += loss
                val_loss_sub[0] += loss_xy.item()
                val_loss_sub[1] += loss_wh.item()
                val_loss_sub[2] += loss_obj.item()
                val_loss_sub[3] += loss_noobj.item()
                val_loss_sub[4] += loss_class.item()

                batch_cnt+=1

            val_loss = val_loss.item() / batch_cnt
            val_loss_sub /= batch_cnt

        # tensorboard record
        writer.add_scalar('val_loss/loss', val_loss, epoch + 1)
        writer.add_scalar('val_loss/loss_xy', val_loss_sub[0].item(), epoch + 1)
        writer.add_scalar('val_loss/loss_wh', val_loss_sub[1].item(), epoch + 1)
        writer.add_scalar('val_loss/loss_obj', val_loss_sub[2].item(), epoch + 1)
        writer.add_scalar('val_loss/loss_noobj', val_loss_sub[3].item(), epoch + 1)
        writer.add_scalar('val_loss/loss_class', val_loss_sub[4].item(), epoch + 1)

        # save last ckpt
        ckpt = {'model': {k: v for k, v in model.state_dict().items() if k in save_names}, # only save trainable params
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}
        torch.save(ckpt, ckpt_path)

        # save best ckpt
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = {'model': {k: v for k, v in model.state_dict().items() if k in save_names}, # only save trainable params
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss}

            torch.save(best_ckpt, best_ckpt_path)

            # save best ckpt for submission (model params only)
            best_ckpt_for_submission = best_ckpt['model']
            torch.save(best_ckpt_for_submission, best_ckpt_path.replace('.pth', '_submission.pth'))

        # print loss
        print("train loss : {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(train_loss_sub[0].item(), train_loss_sub[1].item(),
                                                                        train_loss_sub[2].item(), train_loss_sub[3].item(),
                                                                        train_loss_sub[4].item()))
        print("val   loss : {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(val_loss_sub[0].item(), val_loss_sub[1].item(),
                                                                        val_loss_sub[2].item(), val_loss_sub[3].item(),
                                                                        val_loss_sub[4].item()))
        print("Total train loss : {:.5f} | Total val loss : {:.5f} | Best loss : {:.5f} \n".format(train_loss, val_loss, best_loss))





if __name__ == '__main__':

    model = Yolo(grid_size, num_boxes, num_classes)
    model = model.to(device)
    pretrained_weights = torch.load(pretrained_backbone_path)
    model.load_state_dict(pretrained_weights)

    # Freeze the backbone network.
    model.features.requires_grad_(False)
    model_params = [v for v in model.parameters() if v.requires_grad is True]
    optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # Load the last checkpoint if exits.
    ckpt_path = os.path.join(ckpt_dir.replace("\\", ""), 'last.pth') 

    if os.path.exists(ckpt_path): 
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        last_epoch = ckpt['epoch'] + 1
        print('Last checkpoint is loaded. start_epoch:', last_epoch)
    else:
        print('No checkpoint is found.')

    compute_loss = Loss(device, grid_size, num_boxes, num_classes)

    os.makedirs(ckpt_dir)

    train()