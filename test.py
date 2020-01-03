from mylib.models import densesharp, metrics, losses,densenet
from mylib.utils.misc import crop
from keras.models import load_model
from keras.optimizers import Adamax,Adam
import numpy as np
import csv
from tqdm import tqdm

#指定数据存放路径为PATH(注意加一个/)
DATA_PATH='data/test/'
index_test = tuple(np.arange(117))
name_test = []


voxel_test = []
seg_test = []
for i in tqdm(range(584), desc='reading_test_data'):
    try:
        data = np.load(DATA_PATH+'candidate{}.npz'.format(i))	#依次读取测试数据中的candidate{i}文件
    except FileNotFoundError:									#无该文件时直接进入下一次循环
        continue
    try:
        voxel_test = np.append(voxel_test, np.expand_dims(data['voxel'], axis=0), axis=0)
        seg_test = np.append(seg_test, np.expand_dims(data['seg'], axis=0), axis=0)
        name_test.append('candidate{}.npz'.format(i))
    except ValueError:
        voxel_test = np.expand_dims(data['voxel'], axis=0)
        seg_test = np.expand_dims(data['seg'], axis=0)
        name_test.append('candidate{}.npz'.format(i))
        
        
        
seg_test = seg_test.astype(int)         #将seg布尔array转换为1/0整数
X_test_sharp=voxel_test 
X_test_net= voxel_test*seg_test          #将结节抠出来



X_test_sharp=X_test_sharp.astype(np.float32)
X_test_net=X_test_net.astype(np.float32)
training_test_size = X_test_sharp.shape[0]  #训练数据集的数量
X_test_new=crop(X_test_sharp[0],(50,50,50),(32,32,32))
X_test_new2=crop(X_test_net[0],(50,50,50),(32,32,32))
X_test_new=np.expand_dims(X_test_new,axis=0)
X_test_new2=np.expand_dims(X_test_new2,axis=0)
print(X_test_new.shape) 
print(X_test_new2.shape) 
test_batch_size = X_test_sharp.shape[0]
for i in tqdm(range(test_batch_size-1),desc='croping'):
    X_test_new=np.append(X_test_new,np.expand_dims(crop(X_test_sharp[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
    
for i in tqdm(range(test_batch_size-1),desc='croping'):
    X_test_new2=np.append(X_test_new2,np.expand_dims(crop(X_test_net[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
  
del X_test_sharp
del X_test_net

X_test_new = X_test_new.reshape(X_test_new.shape[0], 32, 32, 32, 1)     #将训练数据集整合成5d张量
X_test_new2 = X_test_new2.reshape(X_test_new2.shape[0], 32, 32, 32, 1) #将训练数据集整合成5d张量
print(X_test_new.shape)
print(X_test_new2.shape)

#加载DenseSharp的模型
model = densesharp.get_compiled(output_size=2,
                                optimizer=Adamax(1.e-3),
                                loss={"clf": 'categorical_crossentropy',
                                      "seg": losses.DiceLoss()},
                                metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                 metrics.invasion_acc, metrics.invasion_fmeasure,
                                                 metrics.invasion_precision, metrics.invasion_recall,
                                                 metrics.ia_acc, metrics.ia_fmeasure,
                                                 metrics.ia_precision, metrics.ia_recall],
                                         'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                loss_weights={"clf": 1., "seg": 0.2},
                                weight_decay=0)
print('sharp_load_down!')
model = load_model('model_sharp.h5',custom_objects={ 'dice_loss_100':losses.DiceLoss(),'precision': metrics.precision,'recall': metrics.recall,'fmeasure': metrics.fmeasure,
                                              'invasion_acc':metrics.invasion_acc, 'invasion_fmeasure':metrics.invasion_fmeasure,
                                              'invasion_precision':metrics.invasion_precision, 'invasion_recall':metrics.invasion_recall,
                                              'ia_acc':metrics.ia_acc, 'ia_fmeasure':metrics.ia_fmeasure,
                                              'ia_precision':metrics.ia_precision, 'ia_recall':metrics.ia_recall})

y_pred1=model.predict(X_test_new)#*0.66
print('sharp_predict_down!')

#加载DenseNet的模型
model = densenet.get_compiled(optimizer=Adam(lr=1.e-3),
                              loss='categorical_crossentropy',
                              metrics=["categorical_accuracy"])

print('net_load_down!')
model = load_model('model_net.h5')
y_pred2=model.predict(X_test_new2)#*0.65
print('net_predict_down!')

#将数据输出为submission.csv文件
f = open('submission.csv','w',encoding='utf-8',newline='' )
csv_writer = csv.writer(f)
csv_writer.writerow(["Id","Predicted"])
for i in range(117):
    csv_writer.writerow([name_test[i],round(0.5*y_pred1[0][i][1]+0.5*y_pred2[i][1],3)])

f.close()
print('All predict down!')