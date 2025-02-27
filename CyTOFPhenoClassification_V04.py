

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd
import datatable as dt
# from torch.autograd import Variable

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Assuming you have your data prepared in torch.Tensor format
# E.g., x_train, x2_train, y_train are all torch.Tensor

##### Define model #####
class MyModel(nn.Module):
    def __init__(self, input_shape_left, input_shape_right, dropout_prob=0.1, filter_num = 5):
        
        super(MyModel, self).__init__()
        
        self.left_branch = nn.Sequential(
            nn.Conv2d(1, filter_num, (1, input_shape_left[2])),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(),
            # nn.Dropout2d(p=dropout_prob)
            
            nn.Conv2d(filter_num, filter_num, (1, 1)),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(),
            # nn.Dropout2d(p=dropout_prob)
            nn.AvgPool2d((input_shape_left[1], 1)),
            nn.Flatten()
        )

        self.right_branch = nn.Sequential(
            nn.Linear(input_shape_right[-1], 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.merged = nn.Sequential(
            nn.Linear(filter_num + 1, 3),  # The 4 here assumes flattened left branch produces 3 features
            nn.BatchNorm1d(3),
            nn.ReLU(),
            # nn.Dropout2d(p=dropout_prob)
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x_left, x_right):
        left_out = self.left_branch(x_left)
        right_out = self.right_branch(x_right)
        merged = torch.cat((left_out, right_out), dim=1)
        output = self.merged(merged)
        return output


def arcsinh(x):
    return np.arcsinh(x / 5)


###########################################################################################################################################################################################################################################
# ANCHOR - [Parameters Setting]


# set parameters
# change directory

os.chdir("/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/analysis/results/CNN/hcc_benign_health_v04")

# get available GPU
print(torch.cuda.is_available())

# get gpu device name
print(torch.cuda.get_device_name())

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Check if GPU/CUDA is available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


epoch_num = 1000

# set the sample size of each fcs file
sample_size = 20000


predict_column_name = "disease_tag"
label_A = "benign"
label_B = "health"
prefix = "{}_{}_20240729".format(label_A, label_B)

# tune the learning rate to 0.0001 (standard), 0.01
learning_rate = 0.0001

# 设置batch size, batch size很重要，太大得到的Accuracy会很低
batch_size = 20

# set the DataLoader num_worker to 10
num_worker = 10

# set running_test_model to True to test the model
running_test_model = False

outdir = f"/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/analysis/results/CNN/hcc_benign_health_v04/{label_A}_{label_B}"
if not os.path.exists(outdir):
    os.makedirs(outdir)
os.chdir(outdir)

fcs_path = "/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/analysis/results/QC/qc_fcs_tsv"


###############################################################################################################################################################################################################################################################

# '''
hcc_feature_frame = pd.read_csv("/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/raw_data/ready/metadata/liver_final_clinical_information.txt",sep = "\t")

# preprocess the hcc_feature_frame

col = ["ID",'sex', 'age']
hcc_feature_frame = hcc_feature_frame.loc[:,col]

hcc_feature_frame["sex"] = hcc_feature_frame["sex"].replace("1", "Male").replace("2", "Female").replace("男", "Male").replace("女","Female").replace(" 女","Female")

hcc_feature_frame = pd.get_dummies(hcc_feature_frame, columns=["sex"], drop_first=True)
# rename ID to sample_id
hcc_feature_frame = hcc_feature_frame.rename(columns={"ID":"sample_id"})
hcc_feature_frame = hcc_feature_frame[["sample_id","age","sex_Male"]]


# load the /common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/analysis/results/Global/MolecularSubtypes/GlobalAdataProp/HCC_CellProportionFrame.csv, index_col = 0
level2_cell_proportion_frame = pd.read_csv("/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/analysis/results/Global/MolecularSubtypes/GlobalAdataProp/HCC_CellProportionFrame.csv", index_col = 0)


hcc_feature_frame = pd.merge(hcc_feature_frame, level2_cell_proportion_frame, left_on="sample_id", right_index=True, how="left")

# get the columns name and remove the sample_id column
columns_name = hcc_feature_frame.columns.tolist()
columns_name.remove("sample_id")


norm_fun = (lambda x: (x - x.mean()) / x.std())
hcc_feature_frame[columns_name] = hcc_feature_frame[columns_name].transform(norm_fun)

hcc_metadata_frame = pd.read_csv("/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/raw_data/ready/metadata/liver_final_clinical_information.txt",sep = "\t")

hcc_metadata_frame["ID"] = hcc_metadata_frame["ID"].astype(str)

file_name_array = []
for sample_id in hcc_metadata_frame["ID"].tolist():
    if not os.path.exists(os.path.join(fcs_path, f"{sample_id}.tsv.gz")):
        print(f"{sample_id}.tsv.gz does not exist.")
    sample_tsv_path = os.path.join(fcs_path, f"{sample_id}.tsv.gz")
    file_name_array.append(sample_tsv_path)
hcc_metadata_frame["file_name"] = file_name_array

# check the disease_tag count
print(hcc_metadata_frame[predict_column_name].value_counts())

hcc_metadata_frame = (hcc_metadata_frame[hcc_metadata_frame[predict_column_name].isin([label_A, label_B])])[["ID",predict_column_name,"file_name"]]

if running_test_model:
# get top 30 of hcc_metadata_frame to test the model
    # get 30 healthy and 30 liver samples
    frame_array = []
    for disease_tag in hcc_metadata_frame[predict_column_name].unique().tolist():
        sub_frame = hcc_metadata_frame[hcc_metadata_frame[predict_column_name] == disease_tag]
        testing_frame = sub_frame.sample(n=30, random_state=1, replace=False)
        frame_array.append(testing_frame)

    hcc_metadata_frame = pd.concat(frame_array, axis=0)

# get the minimum disease_tag count
minor_disease_tag_count = hcc_metadata_frame[predict_column_name].value_counts().min()
major_disease_tag_count = hcc_metadata_frame[predict_column_name].value_counts().max()

# get training and testing data
training_frame_array = []

testing_frame_array = []

# get the training and testing data    
for disease_tag in [label_A, label_B]:
    sub_frame = hcc_metadata_frame[hcc_metadata_frame[predict_column_name] == disease_tag]
    training_frame = sub_frame.sample(n=int(sub_frame.shape[0] * 0.8), random_state=1, replace=False)
    testing_frame = sub_frame[~sub_frame["file_name"].isin(training_frame["file_name"].tolist())]
    training_frame_array.append(training_frame)
    testing_frame_array.append(testing_frame)

training_frame = pd.concat(training_frame_array, axis=0)
testing_frame = pd.concat(testing_frame_array, axis=0)


logging.info(f"Loading the fcs files and generate the training and testing data")



total_file_num = 0
marker_array = None
file_list = hcc_metadata_frame["file_name"].tolist()
marker_array = ['CD45', 'CD3', 'CD56', 'TCRgd', 'CD196.CCR6.', 'CD14', 'IgD', 'CD123', 'CD85j.ILT2.', 'CD19', 'CD25', 'CD274.PD.L1.', 'CD278.ICOS.', 'CD39', 'CD27', 'CD24', 'CD45RA', 'CD86', 'CD28', 'CD197.CCR7.', 'CD11c', 'CD33', 'CD152.CTLA.4.', 'CD161', 'CD185.CXCR5.', 'CD66b', 'CD183.CXCR3.', 'CD94', 'CD57', 'CD45RO', 'CD127', 'CD279.PD.1.', 'CD38', 'CD194.CCR4.', 'CD20', 'CD16', 'HLA.DR', 'CD4', 'CD8a', 'CD11b']




left_train_frame_array = []
left_test_frame_array = []

right_train_frame_array = []
right_test_frame_array = []


train_metadata_frame_array = []
test_metadata_frame_array = []

fcs_file_array = []
index = 0

for fcs_file in file_list:
    if not os.path.exists(os.path.join(fcs_path, fcs_file)):
        continue
    
    sample_fcs_frame = dt.fread(os.path.join(fcs_path, fcs_file)).to_pandas()
    fcs_file_array.append(fcs_file)
    # print(sample_fcs_frame.shape)

    # get the overlap between the marker_array and the current sample_fcs_frame's columns name
    overlap_marker_array = list(set(marker_array).intersection(set(sample_fcs_frame.columns.tolist())))
    if len(overlap_marker_array) < 40:
        continue

    # Data Augmentation, each fcs file will be sampled 30 times

    if fcs_file in training_frame["file_name"].tolist():
        
        # sample 10000 cells from the sampl_fcs_frame
        subset_sample_fcs_frame = sample_fcs_frame.sample(n=10000, random_state=1, replace=False)

        subset_sample_fcs_frame = subset_sample_fcs_frame[marker_array]
    
        subset_sample_fcs_frame = subset_sample_fcs_frame.iloc[:, 0:40]

        # use arcsinh to transform the data
        subset_sample_fcs_frame = subset_sample_fcs_frame.apply(arcsinh)

        # convert sample_fcs_frame to numpy array
        subset_sample_fcs_frame = np.array(subset_sample_fcs_frame)
        
        sample_metadata_frame = hcc_metadata_frame[hcc_metadata_frame["file_name"] == fcs_file]
        train_metadata_array = np.array(sample_metadata_frame[predict_column_name].tolist())
        train_metadata_array = np.where(train_metadata_array == label_A, 1, 0)

        left_train_frame_array.append(subset_sample_fcs_frame)
        
        # get sample_id
        sample_id = fcs_file.split("/")[-1].split(".")[0]
        # get sub_feature_frame from hcc_feature_frame
        sub_train_feature_frame = hcc_feature_frame[hcc_feature_frame["sample_id"] == sample_id]
        
        # remove sample_id column
        sub_train_feature_frame = sub_train_feature_frame.drop(columns=["sample_id"])
        
        right_train_frame_array.append(sub_train_feature_frame.values)
        train_metadata_frame_array.append(train_metadata_array)
        
    
    if fcs_file in testing_frame["file_name"].tolist():
        
        # sample 10000 cells from the sampl_fcs_frame
        subset_sample_fcs_frame = sample_fcs_frame.sample(n=10000, random_state=1, replace=False)

        subset_sample_fcs_frame = subset_sample_fcs_frame[marker_array]

        subset_sample_fcs_frame = subset_sample_fcs_frame.iloc[:, 0:40]

        # use arcsinh to transform the data
        subset_sample_fcs_frame = subset_sample_fcs_frame.apply(arcsinh)

        # convert sample_fcs_frame to numpy array
        subset_sample_fcs_frame = np.array(subset_sample_fcs_frame)

        # testing data

        test_metadata_frame = testing_frame[testing_frame["file_name"] == fcs_file]
        test_metadata_array = np.array(test_metadata_frame[predict_column_name].tolist())
        # recode test_metadata_array
        test_metadata_array = np.where(test_metadata_array == label_A, 1, 0)

        left_test_frame_array.append(subset_sample_fcs_frame)
        
        # get sample_id
        sample_id = fcs_file.split("/")[-1].split(".")[0]
        # get sub_feature_frame from hcc_feature_frame
        sub_test_feature_frame = hcc_feature_frame[hcc_feature_frame["sample_id"] == sample_id]
        # remove sample_id column
        sub_test_feature_frame = sub_test_feature_frame.drop(columns=["sample_id"])
        
        right_test_frame_array.append(sub_test_feature_frame.values)
        test_metadata_frame_array.append(test_metadata_array)

logging.info("current train sample size: %d" % len(left_train_frame_array))
logging.info("current test sample size: %d" % len(left_test_frame_array))

# get the count of different type of test_metadata_frame_array
print(np.unique(test_metadata_frame_array, return_counts=True))
print(np.unique(train_metadata_frame_array, return_counts=True))

logging.info("Saving to npy files.")
# save left_train_frame_array, left_train_metadata_frame_array, left_test_frame_array, left_test_metadata_frame_array to npy
np.save(f"{prefix}_left_train_frame_array.npy", left_train_frame_array)
np.save(f"{prefix}_left_test_frame_array.npy", left_test_frame_array)

np.save(f"{prefix}_train_metadata_frame_array.npy", train_metadata_frame_array)
np.save(f"{prefix}_test_metadata_frame_array.npy", test_metadata_frame_array)

# save right_train_frame_array, right_train_metadata_frame_array, right_test_frame_array, right_test_metadata_frame_array to npy
np.save(f"{prefix}_right_train_frame_array.npy", right_train_frame_array)
np.save(f"{prefix}_right_test_frame_array.npy", right_test_frame_array)

'''
logging.info("Loading the npy files.")
# load the train_frame_array, train_metadata_frame_array, test_frame_array, test_metadata_frame_array from npy file
left_train_frame_array = np.load(f"{prefix}_left_train_frame_array.npy", allow_pickle=True)
left_test_frame_array = np.load(f"{prefix}_left_test_frame_array.npy", allow_pickle=True)

train_metadata_frame_array = np.load(f"{prefix}_train_metadata_frame_array.npy", allow_pickle=True)
test_metadata_frame_array = np.load(f"{prefix}_test_metadata_frame_array.npy", allow_pickle=True)

right_train_frame_array = np.load(f"{prefix}_right_train_frame_array.npy", allow_pickle=True)
right_test_frame_array = np.load(f"{prefix}_right_test_frame_array.npy", allow_pickle=True)


logging.info("Loading Completed. Start Training the Model...")
'''
#################################################################################################################################################################
# ANCHOR - [convert the train_frame_array to tensor]

left_train_x = torch.tensor(np.array(left_train_frame_array), dtype=torch.float32)
left_test_x = torch.tensor(np.array(left_test_frame_array), dtype=torch.float32)

right_train_x = torch.tensor(np.array(right_train_frame_array), dtype=torch.float32)
right_test_x = torch.tensor(np.array(right_test_frame_array), dtype=torch.float32)


train_y = torch.tensor(np.array(train_metadata_frame_array), dtype=torch.float32)
test_y = torch.tensor(np.array(test_metadata_frame_array), dtype=torch.float32)


# Convert data into DataLoader
left_train_x = left_train_x.unsqueeze(1)
right_train_x = right_train_x.unsqueeze(1)
train_data = TensorDataset(left_train_x, right_train_x, train_y)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_worker)

left_test_x = left_test_x.unsqueeze(1)
right_test_x = right_test_x.unsqueeze(1)
test_data = TensorDataset(left_test_x, right_test_x, test_y)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_worker)

# Initialize the model, criterion, and optimizer
model = MyModel(input_shape_left=left_train_x[0].shape, input_shape_right=right_train_x[0].shape)
model = model.to(device)
criterion = nn.BCELoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# create loss frame
loss_frame = pd.DataFrame(columns=["epoch", "train_loss", "test_loss"])

accuracy_frame = pd.DataFrame(columns=["epoch", "train_accuracy", "test_accuracy"])



for epoch in range(epoch_num):  # Adjust as needed
    
    if epoch % 50 == 0:
        logging.info(f"Epoch {epoch+1}")

    
    model.train()  # set the model to training mode
    model.to(device)
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    # get the training data
    for left_input, right_input, labels in train_loader:
        left_input, right_input = left_input.to(device), right_input.to(device)
        labels = labels.to(device)
        # Proceed with model processing using both inputs

        optimizer.zero_grad()
        outputs = model(left_input, right_input)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        # running_loss += loss.item()

        # Compute accuracy for training data
        predicted = torch.round(outputs.squeeze())  # assuming binary classification; round off to nearest integer
        total_train += labels.size(0)
        correct_train += (predicted == labels.squeeze()).sum().item()

    train_accuracy = 100 * correct_train / total_train
    # get two digits after decimal point
    train_accuracy = float("{:.2f}".format(train_accuracy))
    train_loss = loss.item()
    
    
    
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    # do not compute gradient during validation
    with torch.no_grad():
        for left_inputs, right_input, labels in test_loader:
            left_inputs, right_input, labels = left_inputs.to(device), right_input.to(device), labels.to(device)
            outputs = model(left_inputs, right_input)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            test_loss += loss.item()

            # Compute accuracy for test data
            predicted = torch.round(outputs.squeeze())
            total_test += labels.size(0)
            correct_test += (predicted == labels.squeeze()).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracy = float("{:.2f}".format(test_accuracy))
    
    loss_frame = loss_frame.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss}, ignore_index=True) # type: ignore
    accuracy_frame = accuracy_frame.append({"epoch": epoch, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy}, ignore_index=True) # type: ignore
    
    
    logging.info(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")
    logging.info(f"Epoch {epoch+1}, Training Loss: {train_loss}, Test Loss: {test_loss}")
    
# save the model to pt file
torch.save(model, "model.pt")


# load the model from pt file


# model.eval()  # set the model to evaluation mode


# save the loss_frame and accuracy_frame to csv
loss_frame.to_csv(f"{prefix}_loss.txt", sep="\t", index=False)
accuracy_frame.to_csv(f"{prefix}_accuracy.txt", sep="\t", index=False)

# plot the train and test loss and accuracy

accuracy_frame = pd.read_csv(f"{prefix}_accuracy.txt",sep = "\t")
loss_frame = pd.read_csv(f"{prefix}_loss.txt",sep = "\t")

import matplotlib.pyplot as plt

# Plot the training loss and validation loss
plt.figure()
plt.plot(loss_frame["train_loss"], label="Train Loss")
plt.plot(loss_frame["test_loss"], label="Test Loss")
# plt.plot(accuracy_frame["tra"], label="Train Loss")
# plt.plot(test_loss_array, label="Test Loss")
plt.title("Training and Test Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("train_test_loss.png")
plt.close()  # Close the current plot

# Plot the accuracy curve
plt.figure()
plt.plot(accuracy_frame["train_accuracy"], label="Train Accuracy")
plt.plot(accuracy_frame["test_accuracy"], label="Test Accuracy")
#plt.plot(train_accuracy_array, label="Train Accuracy")
#plt.plot(test_accuracy_array, label="Test Accuracy")
plt.title("Training and Test Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

plt.savefig("train_test_accuracy.png")
plt.close()  # Close the current plot



# Plot the ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# load the model from pt file
final_model = torch.load('model.pt')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# use final_model_weights to initialize the model
y_scores = final_model(left_test_x.to(device),right_test_x.to(device)).cpu().detach().numpy()
fpr, tpr, _ = roc_curve(test_metadata_frame_array, y_scores)

# add training data to the ROC curve
y_scores_train = final_model(left_train_x.to(device),right_train_x.to(device)).cpu().detach().numpy()
fpr_train, tpr_train, _ = roc_curve(train_metadata_frame_array, y_scores_train)

roc_auc = auc(fpr, tpr)
roc_auc_train = auc(fpr_train, tpr_train)

# plot ROC curve
plt.plot(fpr, tpr)
plt.plot(fpr_train, tpr_train)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC = {0:.2f}'.format(roc_auc))
plt.show()
# save to png
plt.savefig("roc_curve.png")





