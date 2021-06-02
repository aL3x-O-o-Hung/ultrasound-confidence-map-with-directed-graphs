import cv2
import numpy as np
from matplotlib import pyplot as plt
from EdgeHistogramSRAD import *
def get_gradient(im,k):
    gradient_map=np.zeros((im.shape[0]-1,im.shape[1],21))
    for i in range(21):
        temp_grad=np.zeros((im.shape[0]-1,im.shape[1],k))
        for j in range(1,k+1):
            temp=np.zeros_like(im)
            if 0<=i<=9:
                ii=10-i
                temp[:-j,:-ii]=im[j:,ii:]
                temp[:-j,-ii:]=np.flip(temp[:-j,-2*ii:-ii])
            elif i==10:
                temp[:-j,:]=im[j:,:]
            else:
                ii=i-10
                temp[:-j,ii:]=im[j:,:-ii]
                temp[:-j,:ii]=np.flip(temp[:-j,ii:2*ii])
            temp_grad[:,:,j-1]=np.abs(im-temp)[:-1,:]
        gradient_map[:,:,i]=np.max(temp_grad,axis=2)
    return gradient_map

def needle_edge(needle):
    needle=(needle>0.9)
    needle=needle.astype(np.float)
    im1=np.zeros_like(needle)
    im2=np.zeros_like(needle)

    im1[:-1,:]=needle[1:,:]
    im2[1:,:]=needle[:-1,:]
    out=((im1-im2)>0.1)
    out=out.astype(np.float)
    return out



def confidence_map(im,var=8,last_row=0.1,alpha=1,down=1,index=1):
    image=sitk.GetImageFromArray(im)
    ESRAD_filter=EdgeEnhancementHistogramMatchingSRADFilter()
    rect=[50,50,10,10]
    ESRAD_filter.SetROI(rect)
    ESRAD_filter.SetNumberOfIterations(3000)
    ESRAD_filter.SetTimeStep(0.1)
    ESRAD_filter.SetMaximumError(0.000001)
    ESRAD_filter.SetEdgeDiffusionScale(0.05,down)
    filted_image=ESRAD_filter.Execute(image)
    im=sitk.GetArrayFromImage(filted_image)
    t=np.linspace(-10,10,21)
    bump=np.exp(-0.5/var*t**2)
    gauss=bump/np.trapz(bump)
    gauss=np.reshape(gauss,(1,21))
    h=im.shape[0]
    tempp=np.zeros((im.shape[0]-1,1))
    for i in range(1,im.shape[0]):
        tempp[i-1,0]=np.exp(-alpha*(i+1)/im.shape[0])
    #maxx=np.max(tempp)
    #tempp/=maxx
    #print(tempp)
    #x=-np.log(last_row)/h
    x=-np.log(last_row)/np.sum(tempp,axis=0)[0]
    gradient_map=get_gradient(im,1)
    gradient_map+=0.0001
    mean=np.mean(gradient_map,axis=1)
    mean=np.reshape(mean,(gradient_map.shape[0],1,gradient_map.shape[2]))
    tempp=np.reshape(tempp,(tempp.shape[0],1,1))
    coefficient=np.exp(-(gradient_map/mean)**index*x*tempp)
    confidence=np.zeros_like(im)
    confidence[0,:]=1
    for i in range(1,confidence.shape[0]):
        temp=np.zeros(confidence.shape[1]+20)
        temp[10:-10]=confidence[i-1,:]
        temp[:10]=temp[10:20][::-1]
        temp[-10:]=temp[-20:-10][::-1]
        coef=coefficient[i-1,:,:]*gauss
        coef=coef.transpose()
        temp_matrix=np.zeros_like(coef)
        for j in range(temp_matrix.shape[1]):
            temp_matrix[:,j]=temp[j:j+21]
        temp_matrix*=coef
        confidence[i,:]=np.sum(temp_matrix,axis=0)
    return confidence


def confidence_map_needle_artifact(im,var=8,last_row=0.1,alpha=1,down=1,index=1):
    from hpu import load
    needle,artifact=load(im)
    needle_e=needle_edge(needle)
    image=sitk.GetImageFromArray(im)
    ESRAD_filter=EdgeEnhancementHistogramMatchingSRADFilter()
    rect=[50,50,10,10]
    ESRAD_filter.SetROI(rect)
    ESRAD_filter.SetNumberOfIterations(500)
    ESRAD_filter.SetTimeStep(0.1)
    ESRAD_filter.SetMaximumError(0.000001)
    ESRAD_filter.SetEdgeDiffusionScale(0.05,down)
    filted_image=ESRAD_filter.Execute(image)
    im=sitk.GetArrayFromImage(filted_image)
    t=np.linspace(-10,10,21)
    bump=np.exp(-0.5/var*t**2)
    gauss=bump/np.trapz(bump)
    gauss=np.reshape(gauss,(1,21))
    tempp=np.zeros((im.shape[0]-1,1))
    for i in range(1,im.shape[0]):
        tempp[i-1,0]=np.exp(-alpha*(i+1)/im.shape[0])
    #maxx=np.max(tempp)
    #tempp/=maxx
    #print(tempp)
    #x=-np.log(last_row)/h
    x=-np.log(last_row)/np.sum(tempp,axis=0)[0]
    gradient_map=get_gradient(im,1)
    gradient_map+=0.0001
    mean=np.mean(gradient_map,axis=1)
    mean=np.reshape(mean,(gradient_map.shape[0],1,gradient_map.shape[2]))
    tempp=np.reshape(tempp,(tempp.shape[0],1,1))
    relative=gradient_map/mean
    mm=np.max(gradient_map)
    xxx=np.argwhere(needle_e>0.5)
    for xx in xxx:
        for i in range(relative.shape[2]):
            relative[xx[0],xx[1],i]=mm/mean[xx[0],0,i]
    xxx=np.argwhere(artifact>0.05)
    for xx in xxx:
        for i in range(relative.shape[2]):
            relative[xx[0],xx[1],i]=1
    coefficient=np.exp(-(relative)**index*x*tempp)
    confidence=np.zeros_like(im)
    confidence[0,:]=1
    for i in range(1,confidence.shape[0]):
        temp=np.zeros(confidence.shape[1]+20)
        temp[10:-10]=confidence[i-1,:]
        temp[:10]=temp[10:20][::-1]
        temp[-10:]=temp[-20:-10][::-1]
        coef=coefficient[i-1,:,:]*gauss
        coef=coef.transpose()
        temp_matrix=np.zeros_like(coef)
        for j in range(temp_matrix.shape[1]):
            temp_matrix[:,j]=temp[j:j+21]
        temp_matrix*=coef
        confidence[i,:]=np.sum(temp_matrix,axis=0)
    return confidence*(1-artifact)


def structural_confidence_map(im,ref_conf,var=8,last_row=0.1,alpha=1,down=1,index=1):
    ref=np.max(ref_conf,axis=1)
    image=sitk.GetImageFromArray(im)
    ESRAD_filter=EdgeEnhancementHistogramMatchingSRADFilter()
    rect=[50,50,10,10]
    ESRAD_filter.SetROI(rect)
    ESRAD_filter.SetNumberOfIterations(3000)
    ESRAD_filter.SetTimeStep(0.1)
    ESRAD_filter.SetMaximumError(0.000001)
    ESRAD_filter.SetEdgeDiffusionScale(0.05,down)
    filted_image=ESRAD_filter.Execute(image)
    im=sitk.GetArrayFromImage(filted_image)
    t=np.linspace(-10,10,21)
    bump=np.exp(-0.5/var*t**2)
    gauss=bump/np.trapz(bump)
    gauss=np.reshape(gauss,(1,21))
    h=im.shape[0]
    tempp=np.zeros((im.shape[0]-1,1))
    for i in range(1,im.shape[0]):
        tempp[i-1,0]=np.exp(-alpha*(i+1)/im.shape[0])
    #maxx=np.max(tempp)
    #tempp/=maxx
    #print(tempp)
    #x=-np.log(last_row)/h
    x=-np.log(last_row)/np.sum(tempp,axis=0)[0]
    gradient_map=get_gradient(im,1)
    gradient_map+=0.0001
    mean=np.mean(gradient_map,axis=1)
    mean=np.reshape(mean,(gradient_map.shape[0],1,gradient_map.shape[2]))
    tempp=np.reshape(tempp,(tempp.shape[0],1,1))
    coefficient=np.exp(-(gradient_map/mean)**index*x*tempp)
    confidence=np.zeros_like(im)
    confidence[0,:]=1
    for i in range(1,confidence.shape[0]):
        temp=np.zeros(confidence.shape[1]+20)
        temp[10:-10]=confidence[i-1,:]
        temp[:10]=temp[10:20][::-1]
        temp[-10:]=temp[-20:-10][::-1]
        coef=coefficient[i-1,:,:]*gauss
        coef=coef.transpose()
        temp_matrix=np.zeros_like(coef)
        for j in range(temp_matrix.shape[1]):
            temp_matrix[:,j]=temp[j:j+21]
        temp_matrix*=coef
        confidence[i,:]=np.sum(temp_matrix,axis=0)
        for j in range(confidence.shape[1]):
            if confidence[i,j]>ref[i]:
                confidence[i,j]=ref[i]
    ref=np.reshape(ref,(ref.shape[0],1))
    return confidence/ref




def confidence_map_without_smoothing(im,var=8,last_row=0.1,alpha=1,down=1,index=1):

    t=np.linspace(-10,10,21)
    bump=np.exp(-0.5/var*t**2)
    gauss=bump/np.trapz(bump)
    gauss=np.reshape(gauss,(1,21))
    h=im.shape[0]
    tempp=np.zeros((im.shape[0]-1,1))
    for i in range(1,im.shape[0]):
        tempp[i-1,0]=np.exp(-alpha*(i+1)/im.shape[0])
    #maxx=np.max(tempp)
    #tempp/=maxx
    #print(tempp)
    #x=-np.log(last_row)/h
    x=-np.log(last_row)/np.sum(tempp,axis=0)[0]
    gradient_map=get_gradient(im,1)
    gradient_map+=0.0001
    mean=np.mean(gradient_map,axis=1)
    mean=np.reshape(mean,(gradient_map.shape[0],1,gradient_map.shape[2]))
    tempp=np.reshape(tempp,(tempp.shape[0],1,1))
    coefficient=np.exp(-(gradient_map/mean)**index*x*tempp)
    confidence=np.zeros_like(im)
    confidence[0,:]=1
    for i in range(1,confidence.shape[0]):
        temp=np.zeros(confidence.shape[1]+20)
        temp[10:-10]=confidence[i-1,:]
        temp[:10]=temp[10:20][::-1]
        temp[-10:]=temp[-20:-10][::-1]
        coef=coefficient[i-1,:,:]*gauss
        coef=coef.transpose()
        temp_matrix=np.zeros_like(coef)
        for j in range(temp_matrix.shape[1]):
            temp_matrix[:,j]=temp[j:j+21]
        temp_matrix*=coef
        confidence[i,:]=np.sum(temp_matrix,axis=0)
    return confidence






'''
from confidence_map import confidence_map2d

def paint(conf,x1,y1,x2,y2,color):
    for i in range(3):
        conf[x1:x2,y1:y1+2,i]=color[i]
        conf[x1:x2,y2-2:y2,i]=color[i]
        conf[x1:x1+2,y1:y2,i]=color[i]
        conf[x2-2:x2,y1:y2,i]=color[i]
    return conf

#crop=[89,624,178,678]
crop=[96,624,178,678]
ref=cv2.imread('../data/temp/6/56.png')
ref=cv2.cvtColor(ref[crop[0]:crop[1],crop[2]:crop[3],:],cv2.COLOR_BGR2GRAY)/255.0
ref=confidence_map(ref)
#aa=[125,85,86]
aa=[125,86]
kk=0
for i in aa:
    print(i)
    path='../data/needle/training/'+str(i)+'.png'
    im=cv2.imread(path)
    x=cv2.cvtColor(im[crop[0]:crop[1],crop[2]:crop[3],:],cv2.COLOR_BGR2GRAY)/255.0
    conf=confidence_map(x)
    stru=structural_confidence_map(x,ref)
    temp_conf=confidence_map2d(x,alpha=1,beta=90,gamma=0.01,spacing=None,solver_mode='bf')
    temp=[temp_conf,conf,stru]
    a=np.zeros((x.shape[0]+2,x.shape[1]+2,3))
    a[1:-1,1:-1,0]=x
    a[1:-1,1:-1,1]=x
    a[1:-1,1:-1,2]=x
    if i==125:
        a=paint(a,1,170,12,220,[0,0,1])
        ttt=a[1:12,170:220,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        cv2.imwrite('../isbi_paper_confidence/00_1.png',ttt*255)
        a=paint(a,30,185,85,235,[0,1,0])
        ttt=a[30:85,185:235,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        cv2.imwrite('../isbi_paper_confidence/00_2.png',ttt*255)
        a=paint(a,30,245,85,295,[1,1,1])
        ttt=a[30:85,245:295,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        cv2.imwrite('../isbi_paper_confidence/00_3.png',ttt*255)
        a=paint(a,115,150,130,220,[1,1,0])
        ttt=a[115:130,150:220,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        cv2.imwrite('../isbi_paper_confidence/00_4.png',ttt*255)
        a=paint(a,195,165,230,205,[1,0,1])
        ttt=a[195:230,165:205,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        ttt=ttt**0.9
        cv2.imwrite('../isbi_paper_confidence/00_5.png',ttt*255)
        a=paint(a,195,220,230,260,[0,1,1])
        ttt=a[195:230,220:260,:]
        ttt=cv2.resize(ttt,(ttt.shape[1]*10,ttt.shape[0]*10),interpolation=cv2.INTER_NEAREST)
        o=np.zeros((ttt.shape[0],1,3))
        ttt=np.concatenate((o,ttt,o),axis=1)
        o=np.zeros((1,ttt.shape[1],3))
        ttt=np.concatenate((o,ttt,o),axis=0)
        cv2.imwrite('../isbi_paper_confidence/00_6.png',ttt*255)
        a=a[:,100:,:]
    else:
        a=a[:,:-100,:]
    for t in temp:
        b=np.zeros((x.shape[0]+2,x.shape[1]+2-100,3))
        if i==125:
            tt=t[:,100:]
        else:
            tt=t[:,:-100]
        b[1:-1,1:-1,0]=tt
        b[1:-1,1:-1,1]=tt
        b[1:-1,1:-1,2]=tt
        o=np.ones((x.shape[0]+2,5,3))
        a=np.concatenate((a,o,b),axis=1)
    if kk==0:
        c=a
        kk+=1
    else:
        o=np.ones((5,c.shape[1],3))
        c=np.concatenate((c,o,a),axis=0)

cv2.imwrite('../isbi_paper_confidence/00_.png',c*255)
'''


