import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
import copy
import torch.nn.functional as F
from skimage import morphology
import cv2
from scipy.signal import find_peaks

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def dice_coeff(input,target,epslion = 1e-7):
    if input.dim()==2:
        inter = torch.dot(input.reshape(-1),target.reshape(-1))
        sets_sum = torch.sum(input)+torch.sum(target)
        if sets_sum.item()==0:
            sets_sum = 2*inter
        return (2*inter+epslion) /(sets_sum+epsilon)
    else:
        # compute dice score
        dims = (1,2,3)
        inter = torch.sum(input * target, dims)
        cardi = torch.sum(input + target, dims)
        dice = (2. *inter +epslion)/(cardi+epslion)
        return torch.mean(dice) 
def dice_loss(input,target):
    assert input.size()==target.size()
    return 1-dice_coeff(input,target)


def keep_ratio_resize(img,size=572):
    '''
    this is to resize the image and also keep ratio for image.
    '''
    w,h = img.size
    new_image = np.zeros((size,size),dtype=np.uint8)
    if h>w:
        aspect_ratio = w/h
        new_width = int(size*aspect_ratio)
        resize_img = np.array(img.resize((new_width,size)))
        y_start = int((size-new_width) / 2)
        new_image[:,y_start:y_start+new_width] = resize_img
    else:
        aspect_ratio = h/w
        new_height = int(size*aspect_ratio)
        resize_img = np.array(img.resize((size,new_height)))
        x_start = int((size-new_height) / 2)
        new_image[x_start:x_start+new_height,:] = resize_img
    return Image.fromarray(new_image)

def get_skeleton(image,trunk = 0.02):
    w,h = image.shape
    row_sum = image.sum(axis=1)
    skeleton = np.zeros_like(image)
    points_x = []
    points_y = []
    for row in range(w):
        if row_sum[row]>0:
            row_now = image[row]
            first_none_zero = row_now.nonzero()[0].min()
            last_none_zero = row_now.nonzero()[0].max()
            x = row
            y = int((first_none_zero + last_none_zero)/2)
            points_x.append(x)
            points_y.append(y)
            skeleton[x,y]=1
            
    # trunck the starting and ending error bar
    start_index = int(len(points_x)*trunk)
    end_index = int(len(points_x)*(1-trunk))
    
    points_x = points_x[start_index:end_index]
    points_y = points_y[start_index:end_index]
    
    skeleton[:points_x[0],:]=0
    skeleton[points_x[-1]:,:]=0
    return skeleton,points_x,points_y

def fit_curve(points_x,points_y, img_ratio=1, box=None,deg = 10, if_draw = False):
    if box is None:
        x_offset = np.min(points_x)
        y_offset = np.min(points_y)
    else:
        x_offset = np.array(box[1],dtype=int)
        y_offset = np.array(box[0],dtype=int)
    
    
    # remove the offset
    # resize to 0-572
    x_fit = (points_x - x_offset) * img_ratio
    y_fit = (points_y - y_offset) * img_ratio
    
    fit_para = np.polyfit(x_fit,y_fit,deg)
    #print('fit_para',fit_para)
    f_x = np.poly1d(fit_para)
    
    y_hat = f_x(x_fit)
    
    if if_draw:
        plt.figure(figsize=(20,8))
        plt.plot(x_fit[:len(y_fit)],y_fit, linewidth=7, label='original curve')
        plt.plot(x_fit,y_hat, linewidth=7,color='darkorange', label = 'fitted curve')
        plt.xlabel('x',fontsize=25)
        plt.ylabel('y',fontsize=25)
        plt.legend(fontsize=25)
        #plt.grid(True)
        plt.show()
    # remove dupilicated x 
    x_fit = [int(i) for i in x_fit]
    x_fit = list(set(x_fit))
    #if box is not None:
    #    x_fit = np.append(x_fit,np.arange(np.max(x_fit)+1,572,dtype=int))
    y_hat = f_x(x_fit)
    return f_x,x_fit,y_hat,x_offset,y_offset

    
    

def find_break_point(f_x,x_fit,tol=50,img_ratio=1,if_draw=False):
    '''
    f_x: the fitted curve function
    '''
    f_x_1d = np.polyder(f_x,1)
    f_x_2d = np.polyder(f_x,2)
    fx_dx = f_x_1d(x_fit)
    peaks, _ = find_peaks(abs(fx_dx))  # highest peaks
    #print(peaks)
    ffx_dx = f_x_2d(x_fit)
    break_i = [0]
    for i in range(len(fx_dx)-1):
        # type 1: the turning point i
        if (fx_dx[i]*fx_dx[i+1]<=0) and (i-break_i[-1]>tol) and (x_fit[-1]-x_fit[i]>tol):
            break_i.append(i)
        # type 2: this part is too long, need to break it.
        elif i-break_i[-1]>200:
            peak_2d = break_i[-1] + 1 + abs(ffx_dx[break_i[-1]+1:i]).argmax()
            fx_peak = fx_dx[peak_2d]
            ffx_peak = ffx_dx[peak_2d]
            #print(i,x_fit[i],break_i[-1],peak_2d,fx_peak,ffx_peak,abs(ffx_dx[break_point[-1]+1:x_fit[i]]).argmax())
            if peak_2d-break_i[-1]>tol and peak_2d<i-tol and abs(fx_peak)<0.3:
                #print(peak_2d,fx_peak,ffx_peak)
                break_i.append(peak_2d)
    break_i.append(len(fx_dx)-1)        
    peaks_select = []
    mid_i = []
    for i in range(len(break_i)-1):
        mid_i.append(int((break_i[i]+break_i[i+1])/2))
        peak_now = peaks[(np.array(x_fit)[peaks]>break_i[i]) & (np.array(x_fit)[peaks]<break_i[i+1])]
        if len(peak_now>0):
            peaks_select.append(peak_now[abs(fx_dx)[peak_now].argmax()])
        
    break_point = [x_fit[i] for i in break_i]
    

    print('break point',break_point)
    print('peaks', peaks)
    if if_draw:
        plt.figure(figsize=(20,8))
        plt.plot(x_fit,f_x(x_fit), color='darkorange',linewidth=7,label = 'fitted curve')
        plt.plot(np.array(break_point), f_x(break_point), 'o',markersize=20,label = 'break-points')
        #plt.plot(np.array(x_fit)[peaks_select], f_x(np.array(x_fit)[peaks_select]), "xr",label='1d-peaks')
        #plt.plot(np.array(x_fit)[mid_i],f_x(np.array(x_fit)[mid_i]), "o",label='mids')
        plt.xlabel('x',fontsize=25)
        plt.ylabel('y',fontsize=25)
        plt.legend(fontsize=25)
        #plt.grid(True)
        plt.figure(figsize=(20,8))
        plt.plot(x_fit,fx_dx,label = '1d-derivative',linewidth=7)
        plt.plot(np.array(x_fit)[peaks], fx_dx[peaks], "xr",markersize=30,label='1d-peaks')
        plt.xlabel('x',fontsize=25)
        plt.ylabel('y',fontsize=25)
        plt.legend(fontsize=25)
        #plt.grid(True)
        plt.figure(figsize=(20,8))
        plt.plot(x_fit,abs(ffx_dx),label='2d-derivative',linewidth=7)
        plt.xlabel('x',fontsize=25)
        plt.ylabel('y',fontsize=25)
        plt.legend(fontsize=25)
        plt.show()
    return break_point,f_x_1d,break_i
    


class mask_angle():
    def __init__(self,image,mask,box=None):
        super(mask_angle, self).__init__()
        self.mask = mask
        self.image = image
        if box is None:
            self.ratio=1
        else:
            self.ratio = 572/(box[3]-box[1])
        self.box = box
    def mask2skeleton(self):  
        #return morphology.skeletonize(self.mask)
        return get_skeleton(self.mask)
    
    def fitskeleton(self,x_fit,y_hat,x_offset,y_offset):
        '''
        fit new skeleton
        '''
        new_skeleton = np.zeros_like(self.mask)
        for i in range(len(x_fit)):
            new_skeleton[int(x_fit[i]/self.ratio)+x_offset,int(y_hat[i]/self.ratio)+y_offset] = 1
        return new_skeleton
    
    def find_break_point(self,points_x,points_y,if_draw=False,tol=35):
        '''
        '''
        f_x, x_fit, y_hat, x_offset, y_offset = fit_curve(points_x,points_y,self.ratio,self.box,if_draw=if_draw)
        break_point_x, f_x_1d, break_i = find_break_point(f_x,x_fit,tol,self.ratio,if_draw = if_draw)
        break_point_x = [int(i/self.ratio + x_offset) for i in break_point_x]
        
        new_skeleton = self.fitskeleton(x_fit,y_hat,x_offset,y_offset)
        return break_point_x, new_skeleton, x_fit, f_x, f_x_1d, x_offset, y_offset,break_i
    
    def sp_skeleton(self,skeleton,break_point_x,tol_ratio=0.25):
        '''
        tol: skip the first tol and last tol part 
        '''
        skes = []
        x_points = list(break_point_x) 
        start_x = []
        end_x =[]
        for i in range(len(x_points)-1):
            ske_new = np.zeros_like(skeleton)
            tol_l = int((x_points[i+1]-x_points[i])*tol_ratio)
            if x_points[i+1]-x_points[i]-2*tol_l<(50/self.ratio): # if the mid part is too short
                tol_l = int((x_points[i+1]-x_points[i])*(tol_ratio-0.05)) 
            #print( np.sum(skeleton[x_points[i]+tol_l:x_points[i+1]-tol_l,:]))
            if np.sum(skeleton[x_points[i]+tol_l:x_points[i+1]-tol_l,:])>20:
                ske_new[x_points[i]+tol_l:x_points[i+1]-tol_l,:] = skeleton[x_points[i]+tol_l:x_points[i+1]-tol_l,:]
                skes.append(ske_new)
                start_x.append(x_points[i])
                end_x.append(x_points[i+1])
        return skes,start_x,end_x
    
    def line_fit(self,mask):
        ptx,pty = np.where(mask>0)
        [vx,vy,x,y] = cv2.fitLine(np.concatenate((ptx.reshape(-1,1),pty.reshape(-1,1)),axis=1), cv2.DIST_L1, 0, 0.01, 0.01)
        return x,y,vx,vy
    
    def draw_center_line(self,image,new_skeleton,thickness=5,color=[0,255,0]):
        thickness = int(thickness/2)
        for row in range(thickness,image.shape[0]-thickness):
            for col in range(image.shape[1]):
                if np.sum(new_skeleton[row-thickness:row+thickness,col-thickness:col+thickness])>0:
                    image[row,col,:] = color
        return image
    
    def steepest_point_method(self,image, new_skeleton, break_points,x_fit,f_x,f_x_1d,x_offset,y_offset,break_i,tol_ratio=0.1,off_l=10):
        '''
        this method calculate the steepest point
        '''
        vxs = []
        vys = []
        xcs = []
        ycs = []
        fx_dx = f_x_1d(x_fit)
        peaks, _ = find_peaks(abs(fx_dx))  # highest peaks
        break_points = (break_points - x_offset)*self.ratio
        
        image = self.draw_center_line(image,new_skeleton)
        for i in range(len(break_points)-1):
            tol_l = (break_points[i+1]-break_points[i])*(tol_ratio)
            peak_now = peaks[(np.array(x_fit)[peaks]>break_points[i]) & (np.array(x_fit)[peaks]<break_points[i+1])]
            if  len(peak_now)==0:
                peak_now = [int((break_i[i]+break_i[i+1])/2)]
            if len(peak_now)>0:
                peak_now = peak_now[abs(fx_dx)[peak_now].argmax()]
                mid = (break_points[i]+break_points[i+1])/2
                off_x = min(int(off_l/np.sqrt(1+fx_dx[peak_now]**2)),int(abs(x_fit[peak_now]-mid)))
                print('off_x',off_x)
                print('peak, mid, tol',x_fit[peak_now],mid,tol_l)
                if x_fit[peak_now]>mid and i<(len(break_points)-2):
                    peak_new = peak_now + off_x # the peak is mainly regulatized by the next tint
                else:
                    peak_new = max(peak_now - off_x,int(tol_l)) # the peak is mainly regulatized by the previous tint
                mean_fx_1d = np.mean(fx_dx[max(peak_new-int(tol_l),0):peak_new+int(tol_l)+1]) # get the mean gradient 
                print(mean_fx_1d)
                vx = 1/np.sqrt(1+mean_fx_1d**2)
                vy = mean_fx_1d/np.sqrt(1+mean_fx_1d**2)
                vxs.append(vx)
                vys.append(vy)
                x_fit_center = x_fit[peak_now]
                y_fit_center = f_x(x_fit_center)
                x_c = int(x_fit_center/self.ratio)+x_offset
                y_c = int(y_fit_center/self.ratio)+y_offset
                xcs.append(x_c)
                ycs.append(y_c)
                l=200
                #print(x_c,y_c)
                x2 = x_c + l*vy
                y2 = y_c + l*(-vx)
                x1 = x_c - l*vy
                y1 = y_c - l*(-vx)
                image = cv2.circle(image, (int(y_c),int(x_c)), radius=int(2/self.ratio)*4, color=(255, 0, 255), thickness=-1)
                image = cv2.line(image, (int(y1), int(x1)), (int(y2), int(x2)),color=(0, 0, 255),thickness=int(2/self.ratio))
        angles = []
        for i in range(len(vxs)-1):
            angles.append(self.get_degree(vxs[i],vys[i],vxs[i+1],vys[i+1]))
        return image,angles,xcs,ycs 
 
                         
    def draw_image_line(self,image,x,y,vx,vy,start_x,end_x,color=(255,0,0),l=100,tol=50):
        l1 = (end_x+tol-x)/vx # x and y are different in image and array
        l2 = (x-start_x-tol)/vx  
        if image.max()<=1:
            image = np.array(image*255,dtype=np.uint8)
        if image.ndim<3:
            image = np.expand_dims(image,0)
        if image.shape[0] ==1:
            # convert gray image into rgb
            image = np.tile(image,(3,1,1)).transpose(1,2,0) 
        x2 = x + l1*vx
        y2 = y + l1*vy
        x1 = x - l2*vx
        y1 = y - l2*vy
        image = image.copy() # Change
        cv2.line(image, (int(y1), int(x1)), (int(y2), int(x2)),color=color,thickness=int(2/self.ratio))
        return image
    
    def find_center_point(self,image,x,y,vx,vy,start_x,end_x,color=(255,0,0)):
        l = ((start_x+end_x)/2-x) /vx 
        x_c = x + l*vx
        y_c = y + l*vy
        #print(x_c,y_c)
        image = image.copy() # Change
        image = cv2.circle(image, (int(y_c),int(x_c)), radius=int(2/self.ratio), color=(0, 0, 255), thickness=-1)
        # draw vertical line (vy,vx) -> vertical (-vx,vy)
        l=200
        x2 = x_c + l*vy
        y2 = y_c + l*(-vx)
        x1 = x_c - l*vy
        y1 = y_c - l*(-vx)
        #image = cv2.circle(
        image = cv2.line(image, (int(y1), int(x1)), (int(y2), int(x2)),color=(0, 0, 255),thickness=int(2/self.ratio))
        return image,x_c,y_c
                         
    def get_degree(self,vx1,vy1,vx2,vy2):
        if len(vx1.shape)>0:
            degree = np.arccos(np.dot([vx1[0],vy1[0]],[vx2[0],vy2[0]]))/np.pi*180
        else:
            degree = np.arccos(np.dot([vx1,vy1],[vx2,vy2]))/np.pi*180
        return degree
                         
    def draw_max_degree(self,image,x_c1,y_c1,x_c2,y_c2,angle,c=255):
        cv2.putText(image,'{:.2f}'.format(angle), (int((y_c1+y_c2)/2+50),(int((x_c1+x_c2)/2))), cv2.FONT_HERSHEY_SIMPLEX, 3, c, 6)
        return image
    