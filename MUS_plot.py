#%%
import os
import matplotlib.pyplot as plt
import numpy as np
lr0=1e-03
lr1=1e-05
lr2=1e-07
lr3=1e-09

def lrgraph(counter,title,value0,value1,value2,value3,
                             lr0,lr1,lr2,lr3,xlabel):
    plt.figure(counter)
    plt.title(title)
    plt.plot(value0,label=lr0, linestyle='solid')
    plt.plot(value1,label=lr1, linestyle='dashed')
    plt.plot(value2,label=lr2, linestyle='dotted')
    plt.plot(value3,label=lr3, linestyle='dashdot')
    plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.plot(value4,label=lr4, color=color4)  
    plt.legend()
    plt.savefig(r"lr_combined_{}.png".format(title)) 
    plt.show()

def modelgraph(counter,title,value1,value2,value3,
                             lr1,lr2,lr3,xlabel):
    plt.figure(counter)
    plt.title(title)
    plt.plot(value1,label=lr1, linestyle='solid')
    plt.plot(value2,label=lr2, linestyle='dashed')
    plt.plot(value3,label=lr3, linestyle='dotted')
    plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(r"model_combined_{}.png".format(title)) 
    plt.show()

def boxplot4(x1,x2,x3,x4,
            y1,y2,y3,y4,title):
    data=[x1,x2,x3,x4]
    fig,ax=plt.subplots()
    ax.set_title(title)
    ax.set_xticklabels([y1,y2,y3,y4])
    ax.boxplot(data)
    plt.savefig(r"boxplot4_{}.png".format(title))
    plt.show()
def boxplot3(x1,x2,x3,
             y1,y2,y3,title):
    data=[x1,x2,x3]
    fig,ax=plt.subplots()
    ax.set_title(title)
    ax.set_xticklabels([y1,y2,y3])
    ax.boxplot(data)
    plt.savefig(r"boxplot3_{}.png".format(title))
    plt.show()

cwd2=r"C:\Users\Admin\Documents\FYP"
os.chdir(cwd2)
#%%
trainloss1=np.load(r'RESNET18\{}_results\{}_trainloss.npy'.format(lr1,lr1))
validloss1=np.load(r'RESNET18\{}_results\{}_validloss.npy'.format(lr1,lr1))

trainloss2=np.load(r'RESNET18\{}_results\{}_trainloss.npy'.format(lr2,lr2))
validloss2=np.load(r'RESNET18\{}_results\{}_validloss.npy'.format(lr2,lr2))

trainloss3=np.load(r'RESNET18\{}_results\{}_trainloss.npy'.format(lr3,lr3))
validloss3=np.load(r'RESNET18\{}_results\{}_validloss.npy'.format(lr3,lr3))

trainloss0=np.load(r'RESNET18\{}_results\{}_trainloss.npy'.format(lr0,lr0))
validloss0=np.load(r'RESNET18\{}_results\{}_validloss.npy'.format(lr0,lr0))

boxplot4(trainloss0, trainloss1,trainloss2,trainloss3,lr0,lr1,lr2,lr3,"train loss for different lr")
boxplot4(validloss0, validloss1,validloss2,validloss3,lr0,lr1,lr2,lr3,"valid loss for different lr")
boxplot3(validloss1,validloss2,validloss3,lr1,lr2,lr3,"valid loss for different lr (without anomaly)")

lrgraph(1,'train_loss',trainloss0,trainloss1,trainloss2,trainloss3, lr0,lr1,lr2,lr3, "epoch i")
lrgraph(2,'valid_loss',validloss0,validloss1,validloss2,validloss3, lr0,lr1,lr2,lr3, "iterations j")
#%%
#Print loss graph for different models

resnet18_t_loss = np.load(r"RESNET18\{}_results\{}_trainloss.npy".format(lr1,lr1))
resnet18_v_loss = np.load(r"RESNET18\{}_results\{}_validloss.npy".format(lr1,lr1))

restnet50_t_loss = np.load(r"RESNET50\{}_results\{}_trainloss.npy".format(lr1,lr1))
restnet50_v_loss = np.load(r"RESNET50\{}_results\{}_validloss.npy".format(lr1,lr1))

alexnet_t_loss =np.load(r"ALEXNET\{}_results\{}_trainloss.npy".format(lr1,lr1))
alexnet_v_loss =np.load(r"ALEXNET\{}_results\{}_validloss.npy".format(lr1,lr1))

boxplot3(resnet18_t_loss,restnet50_t_loss,alexnet_t_loss,
        "resnet18","resnet50","alexnet","train loss for different model")
boxplot3(resnet18_v_loss,restnet50_v_loss,alexnet_v_loss,
            "resnet18","resnet50","alexnet","valid loss for different model")

modelgraph(1,"train_loss",resnet18_t_loss,restnet50_t_loss,alexnet_t_loss,
                                    "resnet18","resnet50","alexnet",
                                    "epoch i")
modelgraph(2,"valid_loss",resnet18_v_loss,restnet50_v_loss,alexnet_v_loss,
                                    "resnet18","resnet50","alexnet",
                                    "iterations j")
#%%
#Print accuracy graph for different models
resnet18_t_acc = np.load(r"RESNET18\{}_results\{}_trainacc.npy".format(lr1,lr1))
resnet18_v_acc = np.load(r"RESNET18\{}_results\{}_validacc.npy".format(lr1,lr1))

restnet50_t_acc = np.load(r"RESNET50\{}_results\{}_trainacc.npy".format(lr1,lr1))
restnet50_v_acc = np.load(r"RESNET50\{}_results\{}_validacc.npy".format(lr1,lr1))

alexnet_t_acc =np.load(r"ALEXNET\{}_results\{}_trainacc.npy".format(lr1,lr1))
alexnet_v_acc =np.load(r"ALEXNET\{}_results\{}_validacc.npy".format(lr1,lr1))
boxplot3(resnet18_t_acc,restnet50_t_acc,alexnet_t_acc,
        "resnet18","resnet50","alexnet","train acc for different model")
boxplot3(resnet18_v_acc,restnet50_v_acc,alexnet_v_acc,
            "resnet18","resnet50","alexnet","valid acc for different model")
modelgraph(1,"train_accuracy",resnet18_t_acc,restnet50_t_acc,alexnet_t_acc,
                                   "resnet18","resnet50","alexnet",
                                   "epoch i")
modelgraph(2,"valid_accuracy",resnet18_v_acc,restnet50_v_acc,alexnet_v_acc,
                                    "resnet18","resnet50","alexnet",
                                   "iterations j")
#%%
#Print F1 graph for different models
resnet18_t_f1 = np.load(r"RESNET18\{}_results\{}_trainf1.npy".format(lr1,lr1))
resnet18_v_f1 = np.load(r"RESNET18\{}_results\{}_validf1.npy".format(lr1,lr1))

restnet50_t_f1 = np.load(r"RESNET50\{}_results\{}_trainf1.npy".format(lr1,lr1))
restnet50_v_f1 = np.load(r"RESNET50\{}_results\{}_validf1.npy".format(lr1,lr1))

alexnet_t_f1 =np.load(r"ALEXNET\{}_results\{}_trainf1.npy".format(lr1,lr1))
alexnet_v_f1 =np.load(r"ALEXNET\{}_results\{}_validf1.npy".format(lr1,lr1))

boxplot3(resnet18_t_f1,restnet50_t_f1,alexnet_t_f1,
        "resnet18","resnet50","alexnet","training f1 for different model")
boxplot3(resnet18_v_f1,restnet50_v_f1,alexnet_v_f1,
            "resnet18","resnet50","alexnet","valid f1 for different model")
modelgraph(1,"train_F1",resnet18_t_f1,restnet50_t_f1,alexnet_t_f1,
                                    "resnet18","resnet50","alexnet",
                                    "epoch i")
modelgraph(2,"valid_F1",resnet18_v_f1,restnet50_v_f1,alexnet_v_f1,
                                    "resnet18","resnet50","alexnet",
                                    "iterations j")
#%%
# %%
