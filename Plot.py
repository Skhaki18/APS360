import matplotlib.pyplot as plt
import numpy as np


# Plotting the epoch results...

X = np.arange(1,12, 1)
plt.clf()
TrainLoss = [0.6872, 0.6909, 0.4628, 0.4222, 0.3927, 0.3836, 0.3770, 0.3795,0.3757, 0.3730, 0.3720]
TrainClassifier = [0.3092, 0.3093, 0.1343, 0.1207, 0.1092, 0.1047, 0.1039, 0.1056, 0.1042, 0.1029, 0.1024]

TrainBoxReg = [0.1240, 0.1231, 0.1153, 0.1108, 0.1096, 0.1069, 0.1033, 0.1042, 0.1030, 0.1017, 0.1020]
TrainObjectness = [0.1751, 0.1801, 0.1388, 0.1198, 0.1063, 0.1054, 0.1028, 0.1032, 0.1021, 0.1019, 0.1016]
TrainRPNBoxReg = [0.0790 ,0.0784, 0.0744, 0.0709, 0.0676, 0.0667, 0.0669, 0.0665, 0.0665, 0.0665, 0.0660]
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(X,TrainLoss , label = "Train Loss")
ax1.plot(X,TrainClassifier , label = "Train Classifier Loss")
ax2.plot(X,TrainBoxReg , label = "Train Box Reg Loss")
ax2.plot(X,TrainObjectness , label = "Train Objectness Loss")
ax2.plot(X,TrainRPNBoxReg , label = "Train RPN Box Reg Loss")

ax1.set(xlabel='Epochs', ylabel='Loss', title='Plotting Train Loss')
ax1.legend()
ax2.set(xlabel='Epochs', ylabel='Loss', title='Plotting Train Loss')
ax2.legend()
fig.set_size_inches(14, 8)
fig.savefig('TrainLossResnet.png')

plt.clf()


# lr: 0.005000  loss: 0.5162 (0.6872)  loss_classifier: 0.1572 (0.3092)  loss_box_reg: 0.1269 (0.1240)  loss_objectness: 0.1371 (0.1751)  loss_rpn_box_reg: 0.0728 (0.0790)
# lr: 0.005000  loss: 0.4874 (0.6909)  loss_classifier: 0.1507 (0.3093)  loss_box_reg: 0.0964 (0.1231)  loss_objectness: 0.1513 (0.1801)  loss_rpn_box_reg: 0.0654 (0.0784)
# lr: 0.005000  loss: 0.4654 (0.4628)  loss_classifier: 0.1433 (0.1343)  loss_box_reg: 0.1211 (0.1153)  loss_objectness: 0.1257 (0.1388)  loss_rpn_box_reg: 0.0711 (0.0744)
# lr: 0.005000  loss: 0.4075 (0.4222)  loss_classifier: 0.1266 (0.1207)  loss_box_reg: 0.0756 (0.1108)  loss_objectness: 0.1030 (0.1198)  loss_rpn_box_reg: 0.0733 (0.0709)
# lr: 0.000500  loss: 0.4247 (0.3927)  loss_classifier: 0.1198 (0.1092)  loss_box_reg: 0.1202 (0.1096)  loss_objectness: 0.0927 (0.1063)  loss_rpn_box_reg: 0.0527 (0.0676)
# lr: 0.000500  loss: 0.4120 (0.3836)  loss_classifier: 0.1191 (0.1047)  loss_box_reg: 0.1403 (0.1069)  loss_objectness: 0.1109 (0.1054)  loss_rpn_box_reg: 0.0595 (0.0667)
# lr: 0.000500  loss: 0.3669 (0.3770)  loss_classifier: 0.1084 (0.1039)  loss_box_reg: 0.0988 (0.1033)  loss_objectness: 0.0945 (0.1028)  loss_rpn_box_reg: 0.0548 (0.0669)
# lr: 0.000050  loss: 0.3379 (0.3795)  loss_classifier: 0.0808 (0.1056)  loss_box_reg: 0.0654 (0.1042)  loss_objectness: 0.1044 (0.1032)  loss_rpn_box_reg: 0.0680 (0.0665)
# lr: 0.000050  loss: 0.3285 (0.3757)  loss_classifier: 0.0894 (0.1042)  loss_box_reg: 0.0843 (0.1030)  loss_objectness: 0.0903 (0.1021)  loss_rpn_box_reg: 0.0497 (0.0665)
# lr: 0.000050  loss: 0.3871 (0.3730)  loss_classifier: 0.1019 (0.1029)  loss_box_reg: 0.0976 (0.1017)  loss_objectness: 0.0996 (0.1019)  loss_rpn_box_reg: 0.0738 (0.0665)
# lr: 0.000005  loss: 0.3775 (0.3720)  loss_classifier: 0.0917 (0.1024)  loss_box_reg: 0.0781 (0.1020)  loss_objectness: 0.1003 (0.1016)  loss_rpn_box_reg: 0.0644 (0.0660)