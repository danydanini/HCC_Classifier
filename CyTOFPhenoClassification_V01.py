
# Import library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd
import datatable as dt
from torch.autograd import Variable

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Assuming you have your data prepared in torch.Tensor format
# E.g., x_train, x2_train, y_train are all torch.Tensor

##### Define model #####
class MyModel(nn.Module):
    def __init__(self, input_shape, dropout_prob=0.5, filter_num = 3):
        
        super(MyModel, self).__init__()
        
        # (1 x 3) filters, with each filter size of (1, marker #)
        self.conv1 = nn.Conv2d(1, filter_num, (1, input_shape[2]))
        self.bn1 = nn.BatchNorm2d(filter_num)
        # Add dropout layer after conv1 and bn1
        # self.dropout1 = nn.Dropout2d(p=dropout_prob)
        
        # (3 x 3) filters, with each filter size of (1, 1)
        self.conv2 = nn.Conv2d(filter_num, filter_num, (1, 1))
        self.bn2 = nn.BatchNorm2d(filter_num)
        
        # Average of all cells
        self.avgpool = nn.AvgPool2d((input_shape[1], 1))

        # layer 1, 3 neurons
        self.fc1 = nn.Linear(filter_num, filter_num)
        self.bn6 = nn.BatchNorm1d(filter_num)
        
        # layer 2, 3 neurons
        self.fc2 = nn.Linear(filter_num, 1)
        # self.bn7 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        
        # Apply the three additional convolutional layers
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = F.relu(x)
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = F.relu(x)
        
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = F.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        # x = self.bn7(x)
        x = torch.sigmoid(x)
        
        return x

def arcsinh(x):
    return np.arcsinh(x / 5)


###########################################################################################################################################################################################################################################
# ANCHOR - [Main]
if __name__ == '__main__':
    
    # set parameters
    # change directory
    
    print("Current Working Directory " , os.getcwd())
    
    # get available GPU
    print(torch.cuda.is_available())
    
    # get gpu device name
    print(torch.cuda.get_device_name())
    
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Check if GPU/CUDA is available
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    epoch_num = 1000
    
    # set the sample size of each fcs file
    sample_size = 10000
    
    
    predict_column_name = "disease_tag"
    label_A = "benign"
    label_B = "health"
    prefix = "{}_{}".format(label_A, label_B)
    
    # tune the learning rate to 0.0001 (standard), 0.01
    learning_rate = 0.0001

    # 设置batch size, batch size很重要，太大得到的Accuracy会很低
    batch_size = 100

    # set running_test_model to True to test the model
    running_test_model = False
    
    # augment factor for major disease tag
    major_data_augmentation_factor = 2
    
    outdir = f"/common/cuis/projects/IBDNanostring/RawData/scLiver/\
        CyTofPrediction/analysis/results/CNN/hcc_liver_benign_health/{label_A}_{label_B}_{major_data_augmentation_factor}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    os.chdir(outdir)
    
###############################################################################################################################################################################################################################################################
    ### NOTE Data processing: file name standardization, remove corrupted, files, identify hop_id
    
    fcs_path = "/common/cuis/projects/IBDNanostring/RawData/scLiver/CyTofPrediction/fcs_to_tables"
    
    hcc_metadata_frame = pd.read_csv("/common/cuis/projects/IBDNanostring/RawData/\
                                     scLiver/CyTofPrediction/raw_data/ready/metadata/liver_clinical_info.txt",sep = ",")

    hcc_metadata_frame["hop_ID"] = hcc_metadata_frame["hop_ID"].astype(str)

    # get overlap between micronecrosis_frame's ClinicalIndex and hcc_metadata_frame's hop_ID
    # overlap_hop_ID = list(set(micronecrosis_frame["ClinicalIndex"].
    #   tolist()).intersection(set(hcc_metadata_frame["hop_ID"].tolist())))

    file_name_array = []
    for sample_id in hcc_metadata_frame["ID"].tolist():
        if "_B" in sample_id:
            parts = sample_id.split("_B")
            sample_id = parts[0] + "_B" + parts[1].zfill(4) + ".fcs.tsv.gz"
        elif "HC" in sample_id:
            # convert HC001_161125001241 to HC00001_161125001241.fcs.tsv.gz
            sample_id = sample_id[0:2] + "00" + sample_id[2:] + ".fcs.tsv.gz"
        file_name_array.append(sample_id)
    hcc_metadata_frame["file_name"] = file_name_array
    
    # error_fcs = ["L0251A_B0235.fcs.tsv.gz","L0251A_B0236.fcs.tsv.gz","L0251A_B1592.fcs.tsv.gz"]
    # hcc_metadata_frame = hcc_metadata_frame[~hcc_metadata_frame["file_name"].isin(error_fcs)]
    # hcc_metadata_frame = (hcc_metadata_frame[hcc_metadata_frame[
    #   predict_column_name].isin(["liver","health","benign"])])[["ID",predict_column_name,"file_name","age","sex"]]
    hcc_metadata_frame = (hcc_metadata_frame[hcc_metadata_frame[
        predict_column_name].isin([label_A, label_B])])[["ID",predict_column_name,"file_name","age","sex"]]
    hcc_metadata_frame["file_name"] = hcc_metadata_frame["file_name"].drop_duplicates(keep="first")

    ### NOTE select 30 samples per disease class for testing
    #   calculate augmentation factors for balancing class distribution

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
    
    # get the major disease tag
    major_disease_tag = hcc_metadata_frame[predict_column_name].value_counts().idxmax()
    minor_disease_tag = hcc_metadata_frame[predict_column_name].value_counts().idxmin()
    
    # set the data augmentation factor

    logging.info(f"major_disease_tag_count: {major_disease_tag_count}")
    logging.info(f"minor_disease_tag_count: {minor_disease_tag_count}")
    
    major_disease_tag_count = major_disease_tag_count * major_data_augmentation_factor
    # get the ceil of the division
    minor_data_augmentation_factor = int(np.ceil(major_disease_tag_count / minor_disease_tag_count))

    logging.info(f"major_data_augmentation_factor: {major_data_augmentation_factor}")
    logging.info(f"minor_data_augmentation_factor: {minor_data_augmentation_factor}")
    

    ### NOTE Train and test data split
    #   processing flow cytometry data
    #   store markers

    # get training and testing data
    training_frame_array = []
    
    testing_frame_array = []

    # get the training and testing data    
    for disease_tag in hcc_metadata_frame[predict_column_name].unique().tolist():
        sub_frame = hcc_metadata_frame[hcc_metadata_frame[predict_column_name] == disease_tag]
        training_frame = sub_frame.sample(n=int(sub_frame.shape[0] * 0.8), random_state=1, replace=False)
        testing_frame = sub_frame[~sub_frame["file_name"].isin(training_frame["file_name"].tolist())]
        training_frame_array.append(training_frame)
        testing_frame_array.append(testing_frame)

    training_frame = pd.concat(training_frame_array, axis=0)
    testing_frame = pd.concat(testing_frame_array, axis=0)

    logging.info(f"Loading the fcs files and generate the training and testing data")

    reformated_liver_cytof_array = []
    fcs_file_array = []

    total_file_num = 0
    marker_array = None
    file_list = hcc_metadata_frame["file_name"].tolist()
    marker_array = ['CD45', 'CD3', 'CD56', 'TCRgd', 'CD196.CCR6.', 'CD14', 'IgD', 'CD123', 'CD85j.ILT2.', 
                    'CD19', 'CD25', 'CD274.PD.L1.', 'CD278.ICOS.', 'CD39', 'CD27', 'CD24', 'CD45RA', 
                    'CD86', 'CD28', 'CD197.CCR7.', 'CD11c', 'CD33', 'CD152.CTLA.4.', 'CD161', 'CD185.CXCR5.',
                      'CD66b', 'CD183.CXCR3.', 'CD94', 'CD57', 'CD45RO', 'CD127', 'CD279.PD.1.', 'CD38', 
                      'CD194.CCR4.', 'CD20', 'CD16', 'HLA.DR', 'CD4', 'CD8a', 'CD11b']

    accuracy_output_file = open(f"{prefix}_accuracy.txt", "wt")
    accuracy_output_file.write("train_accuracy\ttest_accuracy\n")
    loss_output_file = open(f"{prefix}_loss.txt", "wt")
    loss_output_file.write("train_loss\ttest_loss\n")

    ### NOTE Read flow cytometry data from .fcs files
    #   keep only 40+ markers from each file
    #   augment data

    train_frame_array = []
    train_metadata_frame_array = []
    test_frame_array = []
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
            
            # get the disease tag of the current fcs file
            disease_tag = training_frame[training_frame["file_name"] == fcs_file][predict_column_name].tolist()[0]
            if disease_tag == major_disease_tag:
                for i in range(major_data_augmentation_factor):
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

                    train_metadata_frame_array.append(train_metadata_array)
                    train_frame_array.append(subset_sample_fcs_frame)
            else:
                for i in range(minor_data_augmentation_factor):
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

                    train_metadata_frame_array.append(train_metadata_array)
                    train_frame_array.append(subset_sample_fcs_frame)
        if fcs_file in testing_frame["file_name"].tolist():
            
            # get the disease tag of the current fcs file
            disease_tag = testing_frame[testing_frame["file_name"] == fcs_file][predict_column_name].tolist()[0]
            if disease_tag == major_disease_tag:
                for i in range(major_data_augmentation_factor):
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

                    test_frame_array.append(subset_sample_fcs_frame)
                    test_metadata_frame_array.append(test_metadata_array)
            else:
                for i in range(minor_data_augmentation_factor):
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

                    test_frame_array.append(subset_sample_fcs_frame)
                    test_metadata_frame_array.append(test_metadata_array)


    logging.info("current train sample size: %d" % len(train_frame_array))
    logging.info("current test sample size: %d" % len(test_frame_array))

    # get the count of different type of test_metadata_frame_array
    print(np.unique(test_metadata_frame_array, return_counts=True))
    print(np.unique(train_metadata_frame_array, return_counts=True))

    logging.info("Saving to npy files.")
    # save train_frame_array, train_metadata_frame_array, test_frame_array, test_metadata_frame_array to npy
    np.save(f"{prefix}_train_frame_array.npy", train_frame_array)
    np.save(f"{prefix}_train_metadata_frame_array.npy", train_metadata_frame_array)
    np.save(f"{prefix}_test_frame_array.npy", test_frame_array)
    np.save(f"{prefix}_test_metadata_frame_array.npy", test_metadata_frame_array)
    '''


    logging.info("Loading the npy files.")
    # load the train_frame_array, train_metadata_frame_array, test_frame_array, test_metadata_frame_array from npy file
    train_frame_array = np.load(f"{prefix}_train_frame_array.npy", allow_pickle=True)
    train_metadata_frame_array = np.load(f"{prefix}_train_metadata_frame_array.npy", allow_pickle=True)
    test_frame_array = np.load(f"{prefix}_test_frame_array.npy", allow_pickle=True)
    test_metadata_frame_array = np.load(f"{prefix}_test_metadata_frame_array.npy", allow_pickle=True)


    # get unique values in train_metadata_frame_array and test_metadata_frame_array
    print(f"train_metadata_frame_array unique values: {np.unique(train_metadata_frame_array, return_counts=True)}")
    print(f"test_metadata_frame_array unique values: {np.unique(test_metadata_frame_array, return_counts=True)}")

    '''


    logging.info("Loading Completed. Start Training the Model...")



    #################################################################################################################################################################
    ### NOTE Training step, BCE is used for loss, and Adam is used for optimization
    
    # convert the train_frame_array to tensor
    train_x = torch.tensor(np.array(train_frame_array), dtype=torch.float32)
    train_y = torch.tensor(np.array(train_metadata_frame_array), dtype=torch.float32)
    
    test_x = torch.tensor(np.array(test_frame_array), dtype=torch.float32)
    test_y = torch.tensor(np.array(test_metadata_frame_array), dtype=torch.float32)

    # Convert data into DataLoader
    train_x = train_x.unsqueeze(1)
    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_x = test_x.unsqueeze(1)
    test_data = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize the model, criterion, and optimizer
    model = MyModel(train_x[0].shape) # Transfer the model to GPU
    model = model.to(device)
    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss_array = []
    test_loss_array = []
    
    train_accuracy_array = []
    test_accuracy_array = []


    for epoch in range(epoch_num):  # Adjust as needed
        
        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch+1}")

        
        model.train()  # set the model to training mode
        model.to(device)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # get the training data
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            # convert inputs to cuda tensor

            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                test_loss += loss.item()

                # Compute accuracy for test data
                predicted = torch.round(outputs.squeeze())
                total_test += labels.size(0)
                correct_test += (predicted == labels.squeeze()).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_accuracy = float("{:.2f}".format(test_accuracy))
        test_loss_array.append(test_loss)

        accuracy_output_file.write(f"{train_accuracy}\t{test_accuracy}\n")
        loss_output_file.write(f"{train_loss}\t{test_loss}\n")
        
        logging.info(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")
        logging.info(f"Epoch {epoch+1}, Training Loss: {train_loss}, Test Loss: {test_loss}")
        
    # save the model to pt file
    # torch.save(model.state_dict(), 'model_weights.pt')
    import pickle
    pickle.dump(model, open("model.pkl", "wb"))
    
    # load the model from pt file

    ### NOTE Evaluation
    
    # model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()  # set the model to evaluation mode

    accuracy_output_file.close()
    loss_output_file.close()
    

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











