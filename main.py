#######################################################################
# Paul-Gauthier Noé
# Laboratoire Informatique d'Avignon (LIA), Avignon University, France
# 2021
#######################################################################

import argparse
import numpy as np
import torch
import os
import sys
import math

sys.path.append('utils/')

from torch.utils import data
from torch import distributions as D
from classes import Dataset, RealNVP
from utils import create_partition_dict, create_labels_dict, txt_2_dict


# Here, attribute is binary

max_epochs  = 10                # Nb epochs
alpha       = 0.99              # Adaptation parameter for mu optimisation

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run training, testing or generation')
    parser.add_argument('name_exp',help="forlder name of the experiment", type=str)
    parser.add_argument('xvectors', help="xvectors file", type=str)
    parser.add_argument('list_utt_id',help="list file of the utterances ID", type=str)
    parser.add_argument('att_labels_txt',help="Txt file containing all the spk id and the corresponding att label", type=str)
    parser.add_argument("-m", "--mu", help="Initialisation value for mu, default is 10", nargs='?', const=10, type=float )
    parser.add_argument("-t", "--test", help="Testing forward",action="store_true")
    parser.add_argument("-p", "--prot", help="Protect the data by setting the estimated LLR to zero",action="store_true")
    parser.add_argument("-M", "--model", help="Model name for testing or protection. The model must be stored in name_exp/models/", nargs='?',const='not_given')
    args = parser.parse_args()

    data_file       = args.xvectors

    list_utt_id     = list(np.loadtxt(args.list_utt_id, dtype='str'))
    att_labels_txt  = args.att_labels_txt
    name_exp        = args.name_exp

    #create output folder
    if not os.path.isdir(name_exp):os.mkdir(name_exp)
    if not os.path.isdir(name_exp+"/models"):os.mkdir(name_exp+"/models")

    #CUDA
    use_cuda        = torch.cuda.is_available()
    device          = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Layer dimension
    input_dim   = 512
    hidden_dim  = 2*512

    # Masks
    m1 = np.zeros(input_dim)
    m1[:int(input_dim/2)]=1
    m2 = 1-m1
    masks = torch.from_numpy(np.array([m1,m2] * 3).astype(np.float32))

    #Batch
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

    #Dataset
    partition           = create_partition_dict(list_utt_id, [])
    spk_id_att_labels   = txt_2_dict(att_labels_txt)
    labels              = create_labels_dict(list_utt_id, [], spk_id_att_labels)

    #Generators
    training_set = Dataset(partition['train'], labels, data_file)

    print('dataset created')

    generator      = data.DataLoader(training_set, **params)
    
    print('generator created')

    model       = RealNVP(input_dim, hidden_dim, masks)
    optimizer   = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=1e-4)

    model.to(device)
    
    print('model created')

    MU      = []
    LOSS    = []

    # Training
    if (not args.test) and (not args.prot):
        print('Training')

        mu_init = args.mu
        print("mu_init: "+str(mu_init))
    
        for epoch in range(max_epochs):

            print("_____EPOCH: "+str(epoch+1)+'/'+str(max_epochs)+"_____")
            for i, data in enumerate(generator,0):
                
                local_batch, local_labels = data[0].cuda(), data[1].cuda()
                local_labels = local_labels.view(local_labels.size()[0],1).float()

                # mu update
                if 'mu' in locals():
                    z0 = z[:,0]
                    muMLE   = math.sqrt(((torch.sum(torch.square(z0))).item()/len(z0)) +1)-1
                    mu      = alpha*mu + (1-alpha)*muMLE                            
                else:
                    mu = mu_init

                z, log_detJ = model.f(local_batch)

                # latent class-conditional densities
                m0  = torch.zeros(input_dim)
                m0[0] = mu
                C = torch.eye(input_dim).cuda()
                C[0,0] = 2*mu
                prior0  = D.MultivariateNormal(m0.cuda(), C)
                prior1  = D.MultivariateNormal(-m0.cuda(), C)
                priors   = [prior0, prior1]

                # Log likelihood
                log_p = model.log_prior(z,local_labels,priors)
                loss = -(log_p+log_detJ).mean()

                # Optimise the parameters of the mapping
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
    
                if i % 128 == 0:
                    print("..............")
                    print("Iter %s:" % i, "Loss : %.3f" % loss)
                    print("log_detJ: "+str(log_detJ.mean().item()))
                    print("log_prior: "+str(log_p.mean().item())) 
                    print(mu)
                    LOSS.append(loss.item())
                    MU.append(mu)

            torch.save(model,name_exp+"/models/model_"+str(epoch)+".pt")

        np.savetxt(name_exp+"/loss.txt",np.array(LOSS))
        np.savetxt(name_exp+"/mu.txt",np.array(MU))
   

    # Testing
    if args.test:
        
        print("test")

        if args.model is None:
            print("Model is not provided")
        elif args.model == 'not_given':
            print("Model name is not given, use: -m <model_name>")
        else:
            model       = torch.load(name_exp+"/models/"+args.model)

            model.eval()
            with torch.no_grad():
                Z = []
                ID = []
                I = []
                for i, data in enumerate(generator,0):
                    local_batch, local_labels, samp_id = data[0].cuda(), data[1].cuda(), data[2]
                    local_labels = local_labels.view(local_labels.size()[0],1).float()
                    z, _ = model.f(local_batch)
                    Z = Z + z.tolist()
                    ID = ID + list(samp_id)
                    I = I + local_batch.tolist()

                Z   = np.array(Z)
                ID  = np.array(ID)

                np.savetxt(data_file+".NF.Z",Z,fmt='%.6e')
                np.savetxt(data_file+".NF.Z.ID",ID,fmt='%s')



    # Protection
    if args.prot:

        print("Protection")
        if args.model is None:
            print("Model is not provided")
        elif args.model == 'not_given':
            print("Model name is not given, use: -m <model_name>")
        else:
            model       = torch.load(name_exp+"/models/"+args.model)
            model.eval()
    
            with torch.no_grad():
                X = []
                ID = []
                I = []
                for i, data in enumerate(generator,0):
                    local_batch, local_labels, samp_id = data[0].cuda(), data[1].cuda(), data[2]
                    local_labels = local_labels.view(local_labels.size()[0],1).float()
                    z, _ = model.f(local_batch)
                    z[:,0] = 0
                    x = model.g(z)
                    X = X + x.tolist()
                    ID = ID + list(samp_id)
                    I = I + local_batch.tolist()

                I   = np.array(I)
                ID  = np.array(ID)
                X   = np.array(X)

                np.savetxt(data_file+'.rNVPzLLR',X,fmt='%.6e')
                np.savetxt(data_file+'.rNVPzLLR.ID',ID,fmt='%s')


