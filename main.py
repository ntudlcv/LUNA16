import sys
import os
from data import luna16_dataset
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3dunet.unet3d.model import ResidualUNet3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', default= '')
    parser.add_argument('--epochs', type= int, default= 10)
    parser.add_argument('--lr', type= float, default= 3e-4)
    parser.add_argument('--weight_decay', type= float, default= 0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--bs', help= 'Batch size', type= int, default= 3)
    parser.add_argument('--dataset', default= './p1_data/')
    parser.add_argument('--size_limit', type= int, default= None)
    parser.add_argument('--load', default= None)
    parser.add_argument('--save', default= 'model.pth')
    parser.add_argument('--record', help= 'Record file name', default= 'record/record.txt')
    parser.add_argument('--interval', type= int, help= 'Record interval within an epoch', default= 200)
    args = parser.parse_args()
    
    print('== Training ==')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResidualUNet3D(in_channels=1, out_channels=1).to(device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    train_data = luna16_dataset("train")
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.bs)
    test_data = luna16_dataset("test")
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.bs)
    print(len(train_loader), len(test_loader))
    ### TODO: Define loss function
    
    for epoch in range(args.epochs):
        startTime = time.time()
        model.train()
        train_loss = 0
        best_loss = 1e16
        for i, (data, mask) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            print(output.shape)
            optimizer.zero_grad()
            ### TODO: loss function and backward
            
            
        model.eval()
        with torch.no_grad():
            for i, (data, mask) in enumerate(test_loader):
                data, mask = data.to(device), mask.to(device)
                output = model(data)
                ### TODO: evaluation, save model, etc.
                
            
        
if __name__ == '__main__':
    main()
