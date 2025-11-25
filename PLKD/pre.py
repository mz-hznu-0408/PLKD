import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from .PLKD_model import scTrans_model as create_model
from .PLKD_model import Student

#model_weight_path = "./weights20220429/model-5.pth" 
#mask_path = os.getcwd()+'/mask.npy'

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

def get_weight(att_mat,pathway):
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v = pd.DataFrame(v[0,1:].detach().numpy()).T
    #print(v.size())
    v.columns = pathway
    return v

def prediect(adata,model_weight_path,project,mask_path,laten=False,save_att = 'X_att', save_lantent = 'X_lat',n_step=10000,cutoff=0.1,n_unannotated = 1,batch_size = 50,embed_dim=48,depth=2,num_heads=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_genes = adata.shape[1]
    #mask_path = os.getcwd()+project+'/mask.npy'
    mask = np.load(mask_path)
    project_path = os.getcwd()+'/%s'%project
    pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]
    model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads).to(device)
    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm={}
    for name,parameters in model.named_parameters():
        #print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim),embed_dim,adata.shape[1]))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token,axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns=adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv(project_path+'/gene2token_weights.csv')
    latent = torch.empty([0,embed_dim]).cpu()
    att = torch.empty([0,(len(pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)      
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class,pre_class]
    att = np.c_[att, predict_class,pre_class]
    all_line = adata.shape[0]
    n_line = 0
    adata_list = []
    while (n_line) <= all_line:
        if (all_line-n_line)%batch_size != 1:
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            print(n_line)
            n_line = n_line+n_step
        else:
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            n_line = (all_line-n_line-2)
            print(n_line)
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True)
        with torch.no_grad():
            # predict class
            for step, data in enumerate(data_loader):
                #print(step)
                exp = data
                lat, pre, weights = model(exp.to(device))
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre,1)
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0) 
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= cutoff: 
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        predict_class = np.r_[predict_class,n_c]
                    pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
                l_p = torch.squeeze(lat).cpu().numpy()
                att = torch.squeeze(weights).cpu().numpy()
                meta = np.c_[predict_class,pre_class]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction','Probability']
                meta.index = meta.index.astype('str')
                if laten:
                    l_p = l_p.astype('float32')
                    new = sc.AnnData(l_p, obs=meta)
                else:
                    att = att[:,0:(len(pathway)-n_unannotated)]
                    att = att.astype('float32')
                    varinfo = pd.DataFrame(pathway.iloc[0:len(pathway)-n_unannotated,0].values,index=pathway.iloc[0:len(pathway)-n_unannotated,0],columns=['pathway_index'])
                    new = sc.AnnData(att, obs=meta, var = varinfo)
                adata_list.append(new)
    print(all_line)
    new = ad.concat(adata_list)
    new.obs.index = adata.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(dic)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return(new)

def predict_student(adata, model_weight_path, project, hidden_dims=[256, 128], dropout=0.1, batch_size=50, cutoff=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_path = os.getcwd()+'/%s'%project
    # Use read_csv as it is more standard for csv files, handling index_col=0
    if os.path.exists(project_path+'/label_dictionary.csv'):
        dictionary = pd.read_csv(project_path+'/label_dictionary.csv', index_col=0)
    else:
        raise FileNotFoundError(f"Label dictionary not found at {project_path}/label_dictionary.csv")

    num_classes = len(dictionary)
    # Construct map: index -> label
    # Assuming label_dictionary.csv has index as 0,1,2... and one column with label names
    label_col = dictionary.columns[0]
    label_map = dictionary[label_col].to_dict()
    label_map[num_classes] = 'Unknown'
    
    num_genes = adata.shape[1]
    
    # Initialize model
    model = Student(input_dim=num_genes, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout).to(device)
    
    # Load weights
    if not os.path.exists(model_weight_path):
         raise FileNotFoundError(f"Model weight file not found: {model_weight_path}")
         
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_probs = []
    all_latents = []
    
    n_cells = adata.shape[0]
    # Simple batch processing
    for i in range(0, n_cells, batch_size):
        end = min(i + batch_size, n_cells)
        batch_data = todense(adata[i:end])
        # todense returns matrix or array. Convert to float32 tensor
        batch_tensor = torch.from_numpy(np.array(batch_data)).float().to(device)
        
        with torch.no_grad():
            # Get latent for visualization
            # Note: We need to access encoder part if we want latent. 
            # We added get_latent to Student class.
            latent = model.get_latent(batch_tensor)
            
            # Get logits for prediction
            # We use the head on top of latent in forward, but we can also call head(latent)
            # model(x) calls encoder then head.
            logits = model.head(latent)
            
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()
            
            # Apply cutoff
            unknown_mask = max_probs < cutoff
            preds[unknown_mask] = num_classes
            
            all_latents.append(latent.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(max_probs)
            
    all_latents = np.concatenate(all_latents, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Create obs dataframe
    obs_df = pd.DataFrame({
        'Prediction': all_preds,
        'Probability': all_probs
    }, index=adata.obs_names)
    
    obs_df['Prediction'] = obs_df['Prediction'].map(label_map)
    
    # Merge with original obs
    new_obs = adata.obs.copy()
    new_obs['Prediction'] = obs_df['Prediction']
    new_obs['Probability'] = obs_df['Probability']
    
    # Create new AnnData
    # X is latent space
    new_adata = sc.AnnData(X=all_latents, obs=new_obs)
    
    return new_adata
