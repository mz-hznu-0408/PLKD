import random
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform
import torch.nn.functional as F

from .PLKD_model import scTrans_model as create_model
from .PLKD_model import Student

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)
  
def splitDataSet(adata,label_name='Celltype', tr_ratio= 0.7): 
    """ 
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    #el_data = pd.read_table(data_path,sep=",",header=0,index_col=0)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    el_data = balance_populations(data = el_data)
    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(np.array(valid_dataset)[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(np.array(valid_dataset)[:,-1].astype(np.int64))
    return exp_train, label_train, exp_valid, label_valid, inverse, genes

def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)


def divergence_clustering_loss(logits, embeddings, eps=1e-6):
    """
    Implements Eq.(5): divergence-based clustering loss that encourages class-wise orthogonality.
    """
    logits = logits.contiguous()
    embeddings = embeddings.contiguous()
    batch_size, num_classes = logits.shape
    if batch_size < 2 or num_classes < 2:
        return logits.new_zeros(())

    # Similarity matrix S based on cell embeddings (Eq. definition of s_{i,j}).
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    sim_matrix = torch.exp(-(pairwise_dist ** 2))

    # Y has shape [num_classes, batch]; YS precomputes Y * S for reuse.
    logits_t = logits.transpose(0, 1)
    y_s = torch.matmul(logits_t, sim_matrix)

    loss = logits.new_zeros(())
    for a in range(num_classes - 1):
        ya = logits_t[a]
        ya_s = y_s[a]
        ya_norm = torch.dot(ya_s, ya) + eps
        for b in range(a + 1, num_classes):
            yb = logits_t[b]
            yb_s = y_s[b]
            yb_norm = torch.dot(yb_s, yb) + eps
            numer = torch.dot(ya_s, yb)
            denom = torch.sqrt(ya_norm * yb_norm) + eps
            loss = loss + numer / denom

    return loss / num_classes

def train_one_epoch(model, optimizer, data_loader, device, epoch, divergence_weight=1.0):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss() 
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        latent,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        ce_loss = loss_function(pred, label.to(device))
        div_loss = divergence_clustering_loss(pred, latent) if divergence_weight > 0 else torch.zeros_like(ce_loss)
        loss = ce_loss + divergence_weight * div_loss
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, divergence_weight=1.0):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        latent,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        ce_loss = loss_function(pred, labels.to(device))
        div_loss = divergence_clustering_loss(pred, latent) if divergence_weight > 0 else torch.zeros_like(ce_loss)
        loss = ce_loss + divergence_weight * div_loss
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=8, embed_dim=48,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01, divergence_weight=1.0):
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    #train_weights = os.getcwd()+"/weights%s"%today
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse,genes = splitDataSet(adata,label_name)
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        #print(mask.shape)
        print('Mask loaded!')
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train)+1)
    #print(num_classes)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,drop_last=True)
    model = create_model(num_classes=num_classes, num_genes=len(exp_train[0]),  mask = mask,embed_dim=embed_dim,depth=depth,num_heads=num_heads,has_logits=False).to(device) 
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name) 
    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]  
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                divergence_weight=divergence_weight)
        scheduler.step() 
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch,
                                     divergence_weight=divergence_weight)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if platform.system().lower() == 'windows':
            torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
        else:
            torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
    print('Training finished!')

def self_entropy_loss(logits, eps=1e-6):
    """
    Implements Eq.(8): self entropy loss based on average student logits.
    """
    if logits.ndim != 2:
        raise ValueError("Self entropy expects logits with shape [batch, num_classes].")
    mean_logits = logits.mean(dim=0)
    prob = F.softmax(mean_logits, dim=0)
    entropy = -(prob * torch.log(prob + eps)).sum()
    return entropy / prob.shape[0]


def train_one_epoch_kd(teacher, student, optimizer, data_loader, device, epoch, alpha=0.5, temperature=4.0, self_entropy_weight=1.0):
    teacher.eval()
    student.train()
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction='batchmean')
    
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    optimizer.zero_grad()
    
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        exp = exp.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            _, t_logits, _ = teacher(exp)
            
        s_logits = student(exp)
        
        # KD Loss
        soft_targets = torch.nn.functional.softmax(t_logits / temperature, dim=1)
        log_soft_student = torch.nn.functional.log_softmax(s_logits / temperature, dim=1)
        
        kl_loss = loss_kl(log_soft_student, soft_targets) * (temperature ** 2)
        ce_loss = loss_ce(s_logits, label)
        
        entropy_loss = self_entropy_loss(s_logits) if self_entropy_weight > 0 else torch.zeros_like(kl_loss)
        loss = alpha * kl_loss + (1 - alpha) * ce_loss + self_entropy_weight * entropy_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        accu_loss += loss.detach()
        pred_classes = torch.max(s_logits, dim=1)[1]
        accu_num += torch.eq(pred_classes, label).sum()
        
        data_loader.desc = "[KD train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
        
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_student(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        pred = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[student valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def fit_PLKD(adata, gmt_path, project=None, pre_weights='', label_name='Celltype', max_g=300, max_gs=300, mask_ratio=0.015, n_unannotated=1, batch_size=8, embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10, lrf=0.01, alpha=0.5, temperature=4.0, student_hidden=[256, 128], student_dropout=0.1, divergence_weight=1.0, self_entropy_weight=1.0):
    # Reuse logic from fit_model for setup
    GLOBAL_SEED = 1
    set_seed(GLOBAL_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt','')+'_PLKD_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        
    tb_writer = SummaryWriter(log_dir=project_path)
    
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name)
    
    # Setup Mask for Teacher
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask, pathway = create_pathway_mask(feature_list=genes, dict_pathway=reactome_dict, add_missing=n_unannotated, fully_connected=True)
        
        # Filter mask
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        print('Mask loaded!')

    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    
    num_classes = np.int64(torch.max(label_train)+1)
    
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    # 1. Initialize and Train/Load Teacher
    teacher = create_model(num_classes=num_classes, num_genes=len(exp_train[0]), mask=mask, embed_dim=embed_dim, depth=depth, num_heads=num_heads, has_logits=False).to(device)
    
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        teacher.load_state_dict(preweights_dict, strict=False)
        print('Teacher weights loaded!')
    else:
        print('Training Teacher...')
        # Train Teacher logic (simplified copy from fit_model)
        pg = [p for p in teacher.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
        for epoch in range(epochs): # Using same epochs for teacher as default
            train_loss, train_acc = train_one_epoch(teacher, optimizer, train_loader, device, epoch, divergence_weight=divergence_weight)
            scheduler.step()
            val_loss, val_acc = evaluate(teacher, valid_loader, device, epoch, divergence_weight=divergence_weight)
            # Save teacher if needed, skipping for brevity in this block or save last
        
        torch.save(teacher.state_dict(), project_path+"/teacher_model.pth")
        print('Teacher training finished!')

    # 2. Initialize Student
    student = Student(input_dim=len(exp_train[0]), num_classes=num_classes, hidden_dims=student_hidden, dropout=student_dropout).to(device)
    print('Student initialized!')
    
    # 3. Train Student with KD
    optimizer_s = optim.Adam(student.parameters(), lr=lr) # Adam is usually better for MLP
    # scheduler_s = lr_scheduler.LambdaLR(optimizer_s, lr_lambda=lf) # Reuse schedule?
    
    print('Starting Distillation...')
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_kd(teacher, student, optimizer_s, train_loader, device, epoch, alpha, temperature, self_entropy_weight=self_entropy_weight)
        # scheduler_s.step()
        val_loss, val_acc = evaluate_student(student, valid_loader, device, epoch)
        
        tags = ["student_train_loss", "student_train_acc", "student_val_loss", "student_val_acc"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        
        torch.save(student.state_dict(), project_path+"/student_model-{}.pth".format(epoch))
        
    print('PLKD Training finished!')
