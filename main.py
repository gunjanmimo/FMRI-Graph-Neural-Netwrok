import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import torch.nn.functional as F


from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import networkx as nx
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

# local imports
from models import GCN, GCN_2, GCN_3, GCN_4

# DATA LOADING
basepath = "./data"
basepath_FA = os.path.join(basepath, "FA")
basepath_GM = os.path.join(basepath, "GM")
basepath_RS = os.path.join(basepath, "RS")


CT_CONTROL = -1
CT_RRMS = 0
CT_SPMS = 1
CT_PPMS = 2

# Load demographics
df = pd.read_csv(os.path.join(basepath, "demographics.csv"))

target = df["mstype"].values
# Transform target to 0 -> HV; 1 -> MS
target = target + 1
target[target > 1] = 1


filenames = ["{:04d}.csv".format(x) for x in df["id"]]
data = np.zeros(shape=(len(filenames), 76, 76, 3))

# Load data
for i, filename in enumerate(filenames):
    df = pd.read_csv(os.path.join(basepath_FA, filename), header=None)
    data[i, :, :, 0] = df.values

    df = pd.read_csv(os.path.join(basepath_GM, filename), header=None)
    data[i, :, :, 1] = df.values

    df = pd.read_csv(os.path.join(basepath_RS, filename), header=None)
    data[i, :, :, 2] = df.values

print(data.shape)

# Check proportion of pwMS
prop = np.where(target == 1)[0].shape[0] / target.shape[0]
print("% of pwMS: {:.4f}".format(prop))


# Create graph data
def array_to_graph(data, y, th=0.0,use_edge_index=False):
    
    if use_edge_index:
        
        num_nodes = data.shape[0]
        # Assuming we want to extract node features from 2 layers (other than the connectivity layer)
        node_dim = 2

        # Choose DTI (FA) layer for connectivity; assuming it's the first layer (index 0)
        connectivity_layer = 0

        edge_index = []
        edge_weight = []

        # Edge construction based on DTI (FA) layer
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and data[i, j, connectivity_layer] >= th:
                    edge_index.append([i, j])
                    edge_weight.append(data[i, j, connectivity_layer])

        # Node feature construction from the other layers
        x = np.zeros((num_nodes, node_dim))
        for i in range(num_nodes):
            for layer in range(1, 3):  # Assuming the other two layers are indices 1 and 2
                x[i, layer - 1] = np.sum(data[i, :, layer] >= th)  # Sum of connections as a feature

        # Convert to tensors
        y = torch.tensor([int(y)], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(x, dtype=torch.float)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Create graph data
        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

        return graph_data
    
    
    
    num_nodes = data.shape[0]
    node_dim = 1

    edge_index = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if data[i, j, 0] >= th:
                    edge_index.append([i, j])
                    edge_weight.append(data[i, j, 0])

                if data[i, j, 1] >= th:
                    edge_index.append([i, j])
                    edge_weight.append(data[i, j, 1])

                if data[i, j, 2] >= th:
                    edge_index.append([i, j])
                    edge_weight.append(data[i, j, 2])

    y = torch.tensor([int(y)], dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(np.ones(shape=(num_nodes, node_dim)), dtype=torch.float)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    data = Data(
        x=x, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight, y=y
    )

    return data





### task 3.3 augmentation
def augment_data(data, permutations):
    """
    Augments the data by flipping brain regions based on the provided permutations.

    Parameters:
    - data: A numpy array or a PyTorch tensor of shape (batch_size, channels, height, width).
            Channels correspond to different brain regions.
    - permutations: A list of lists, where each sub-list contains two integers indicating
                    the brain regions to be swapped to perform the flip.

    Returns:
    - augmented_data: A new dataset that includes the original and flipped data, effectively
                      doubling the size of the input dataset.
    """
    # Check the type of the input data and duplicate accordingly
    if isinstance(data, np.ndarray):
        # Use numpy.concatenate for NumPy arrays
        augmented_data = np.concatenate([data, data.copy()], axis=0)
    elif torch.is_tensor(data):
        # Use torch.cat for PyTorch tensors
        augmented_data = torch.cat([data, data.clone()], dim=0)
    else:
        raise TypeError("Data must be either a NumPy array or a PyTorch tensor")

    # Apply the permutations to the second half of the augmented dataset (the cloned part)
    for i, (p1, p2) in enumerate(permutations):
        augmented_data[data.shape[0] :, p1, :, :] = data[:, p2, :, :]
        augmented_data[data.shape[0] :, p2, :, :] = data[:, p1, :, :]

    return augmented_data


permutations = [
    [0, 45],
    [1, 46],
    [2, 47],
    [3, 48],
    [4, 49],
    [5, 50],
    [6, 51],
    [7, 52],
    [8, 53],
    [9, 54],
    [10, 55],
    [11, 56],
    [12, 57],
    [13, 58],
    [14, 59],
    [15, 60],
    [16, 61],
    [17, 62],
    [18, 63],
    [19, 64],
    [20, 65],
    [21, 66],
    [22, 67],
    [23, 68],
    [24, 69],
    [25, 70],
    [26, 71],
    [27, 72],
    [28, 73],
    [29, 74],
    [30, 75],
    [31, 38],
    [32, 39],
    [33, 40],
    [34, 41],
    [35, 42],
    [36, 43],
    [37, 44],
]


def train_model(data, target, device, model, optimizer, loss_fn,use_edge_index):
    total_fold = 2
    skf = StratifiedKFold(n_splits=total_fold)
    NUM_EPOCHS = 20
    preds = np.zeros(data.shape[0])
    fold = 0

    for train_index, test_index in skf.split(data, target):
        fold += 1
        print("Fold: {}".format(fold))

        # split dataset
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
        prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
        print("Train set size     : {}".format(X_train.shape))
        print("Test set size      : {}".format(X_test.shape))
        print("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
        print("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

        # list of Data structures (one for each subject)
        train_graphs = []
        for i in range(X_train.shape[0]):
            g = array_to_graph(X_train[i], y_train[i])
            train_graphs.append(g)

        test_graphs = []
        for i in range(X_test.shape[0]):
            g = array_to_graph(X_test[i], y_test[i],use_edge_index=use_edge_index)
            test_graphs.append(g)



        model = model.to(device)
  

        # train function
        def train():
            model.train()

            train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

            loss_all = 0
            for batch in tqdm(train_loader, total=len(train_loader)):
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                label = batch.y
                label = F.one_hot(label, num_classes=2)
                label = label.type(torch.FloatTensor)
                label = label.to(device)
                loss = loss_fn(output, label)
                loss.backward()
                loss_all += batch.num_graphs * loss.item()
                optimizer.step()

            return loss_all / len(train_graphs)

        # train for N epochs
        for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
            loss_value = train()
            print("Train loss at epoch {}: {:.4f}".format(epoch + 1, loss_value))

        # test phase
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(
                test_graphs, batch_size=len(test_graphs), shuffle=False
            )
            
            for batch in tqdm(test_loader, total=len(test_loader), desc="Test"):
                batch = batch.to(device)
                test_preds = F.softmax(model(batch), dim=1).detach()

        test_preds = test_preds.cpu().numpy()

        test_preds = test_preds[:, 1]
        preds[test_index] = test_preds
        
        
     
        
        auc_roc = roc_auc_score(y_test, test_preds)
        print("Test AUC: {:.2f}".format(auc_roc))

    return preds

#--------------------------------------------------------------------- FINE TUNING -----------------------------------------------------------------------------------#


### task 3.1
available_models = ["GCN", "GCN_2", "GCN_3", "GCN_4"]
available_epochs = [20, 50, 70, 100]
available_optimizers = ["Adam", "SGD", "RMSprop", "Adagrad"]
available_lr = [0.01, 0.001, 0.0001]
available_loss = [
    "CrossEntropy",
    "MSE",
]

# task 3.2: Graph representation and node embeddings
available_use_edge_index = [
    
                             True,    
                            False
                            ]




# task 3.3
available_augmentation = [
                            True, 
                          False
                          ]




def objective(trail):

    
    # task 3.1
    selected_model = trail.suggest_categorical("model", available_models)
    selected_epochs = trail.suggest_categorical("epochs", available_epochs)
    selected_optimizer = trail.suggest_categorical("optimizer", available_optimizers)
    selected_lr = trail.suggest_categorical("lr", available_lr)
    selected_loss = trail.suggest_categorical("loss", available_loss)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if selected_model == "GCN":
        model = GCN()
    elif selected_model == "GCN_2":
        model = GCN_2()
    elif selected_model == "GCN_3":
        model = GCN_3()
    elif selected_model == "GCN_4":
        model = GCN_4()
    model = model.to(device)
    

    # task 3.2
    use_edge_index = trail.suggest_categorical("use_edge_index", available_use_edge_index)
    
    

    # task 3.3 augmentation
    selected_augmentation = trail.suggest_categorical(
        "augmentation", available_augmentation
    )

    # optimizer
    if selected_optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=selected_lr)
    elif selected_optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=selected_lr)
    elif selected_optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=selected_lr)
    elif selected_optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=selected_lr)

    # loss function
    if selected_loss == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    # elif selected_loss == "NLL":
    #     loss_fn = torch.nn.NLLLoss()
    elif selected_loss == "MSE":
        loss_fn = torch.nn.MSELoss()

    if not selected_augmentation:
        preds = train_model(
            data=data,
            target=target,
            device=device,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            use_edge_index=use_edge_index
        )
        final_target = target
    else:
        augmented_data = augment_data(data, permutations)
        augmented_target = np.concatenate([target, target])
        
        
        print("Augmented data size: {}".format(augmented_data.shape))
        print("Augmented target size: {}".format(augmented_target.shape))

        preds = train_model(
            data=augmented_data,
            target=augmented_target,
            device=device,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            use_edge_index=use_edge_index
        )
        # final_target = augmented_target[:len(preds)]
        final_target = augmented_target
        
        


    # Calculate AUC with the adjusted final_target
    auc_roc = roc_auc_score(final_target, preds)

    auc_pr = average_precision_score(final_target, preds)

    best_acc = 0
    best_th = 0
    for th in preds:
        acc = accuracy_score(final_target, (preds >= th).astype(int))
        if acc >= best_acc:
            best_acc = acc
            best_th = th

    print("")
    print("Selected model : {}".format(selected_model))
    print("Selected epochs: {}".format(selected_epochs))
    print("Selected optimizer: {}".format(selected_optimizer))
    print("Selected lr: {}".format(selected_lr))
    print("Selected loss: {}".format(selected_loss))

    prop = np.where(target == 1)[0].shape[0] / target.shape[0]
    print("% of pwMS: {:.4f}".format(prop))
    print("AUC ROC  : {:.4f}".format(auc_roc))
    print("AUC PR   : {:.4f}".format(auc_pr))
    print("ACC      : {:.4f}".format(best_acc))
    print("#" * 50)
    
    

    return best_acc


if __name__ == "__main__":
    # optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    print(study.best_trial.params)
    print(study.best_trial.value)
    print(study.best_trial.number)
    print(study.best_trial.user_attrs)
