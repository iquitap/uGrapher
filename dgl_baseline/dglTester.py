import argparse
import time
import os
from tqdm import *
import torch
from dgl.data import register_data_args
import torch.nn.functional as F
from dataset import *
# from dgl.nn.pytorch.conv import GraphConv
# from dgl.nn.pytorch.conv import SAGEConv
# os.system("export CUDA_VISIBLE_DEVICES=0")

os.environ["PYTHONWARNINGS"] = "ignore"
class DGLTester:
    ''' DGL Test entry '''
    def __init__(self, dataDir, dataset, dim, classes, gpu, training, 
    torch_profile, model, n_epochs, hidden, aggregator_type, num_layers, num_heads, num_outheads,
    in_drop, attn_drop, negative_slope, residual):
        self._dataDir = dataDir
        self._dataset = dataset
        self._dim = dim
        self._classes = classes
        self._gpu = gpu
        self._training = training
        self._torch_profile = torch_profile
        self._model = model
        self._n_epochs = n_epochs
        self._hidden = hidden
        self._aggregator_type = aggregator_type
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_outheads = num_outheads
        self._in_drop = args.in_drop
        self._attn_drop = args.attn_drop
        self._negative_slope = args.negative_slope
        self._residual = args.residual
    
    @property
    def data(self):
        path = os.path.join(self._dataDir, self._dataset+".npz")
        data = custom_dataset(path, self._dim, self._classes, load_from_txt=False)
        return data

    @property
    def cuda(self):
        if self._gpu < 0:
            cuda = False
        else:
            cuda = True
        return cuda

    def main(self):
        print(f'dgl test start')
        data = self.data
        g = data.g

        #if self._model == 'gat':
        #    g = dgl.add_self_loop(g)

        print(f'{g}')
        g = g.int().to(self._gpu)
 
        features = data.x
        labels = data.y
        in_feats = features.size(1)
        n_classes = data.num_classes

        if self._model == 'gat':
            n_edges = g.number_of_edges()
            heads = ([self._num_heads] * self._num_layers) + [self._num_outheads]
        else:
            # normalization
            degs = g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.cuda()
            g.ndata['norm'] = norm.unsqueeze(1)

        if self._model == 'gcn':
            model = GCN(g,
                        in_feats=in_feats,
                        n_hidden=args.hidden,
                        n_classes=n_classes,
                        n_layers=2)
        elif self._model == 'gin':
            model = GIN(g,
                        input_dim=in_feats,
                        hidden_dim=64,
                        output_dim=n_classes,
                        num_layers=5)
        elif self._model == 'sage':
            model = GraphSAGE(g,
                            in_feats,
                            self._hidden,
                            self._classes,
                            self._num_layers,
                            F.relu,
                            self._aggregator_type)
        elif self._model == 'gat':
            model = GAT(g,
                        self._num_layers,
                        in_feats,
                        self._hidden,
                        n_classes,
                        heads,
                        F.elu,
                        self._in_drop,
                        self._attn_drop,
                        self._negative_slope,
                        self._residual)

        if self.cuda: model.cuda()

        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1e-2,
                                     weight_decay=5e-4)
        # initialize graph
        dur = []
        torch.cuda.synchronize()
        def training():
            for epoch in range(args.n_epochs):
                model.train()
                if epoch >= 3:
                    # t0 = time.time()
                    start = time.perf_counter()
                # forward
                logits = model(features)
                # loss = loss_fcn(logits[train_mask], labels[train_mask])
                loss = loss_fcn(logits[:], labels[:])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    torch.cuda.synchronize()
                    dur.append(time.perf_counter() - start)

            if self._model == 'gcn':
                print("DGL Training GCN (L2-H16) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'gin':
                print("DGL Training GIN (L5-H64) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'gat':
                print("DGL Training GAT (L2-H16) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'sage':
                print("DGL Training GraphSage Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            print()

        def inference():
            for epoch in range(args.n_epochs):
                if epoch >= 3:
                    start = time.perf_counter()
                model.eval()
                logits = model(features)
                if epoch >= 3:
                    torch.cuda.synchronize()
                    dur.append(time.perf_counter() - start)

            if self._model == 'gcn':
                print("DGL Inference GCN (L2-H16) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'gin':
                print("DGL Inference GIN (L5-H64) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'gat':
                print("DGL Inference GAT (L2-H16) Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
            elif self._model == 'sage':
                print("DGL Inference GraphSage Time: (ms) {:.3f}".format(np.mean(dur)*1e3))
                
            print()

        if (self._torch_profile == True):
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                if self._training == 'training':
                    training()
                else:
                    inference()
            print(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            if self._training == 'training':
                training()
            else:
                inference()

def str2bool(s):
    if isinstance(s, bool):
        return s
    elif s.lower() in ('true','t','yes','1'):
        return True
    elif s.lower() in ('false','f','n','0'):
        return False
    else:
        print('Unexpected arg value for torch_profile')
        raise argparse.ArgumentTypeError('Unexpected arg value for torch_profile')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # register_data_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )
    parser.add_argument("--dataDir", type=str, default="./dataset/npz/""/home/yjzhou/github/gnn/dataset/osdi-ae-graphs", help="the path to graphs")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
    parser.add_argument("--hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--classes", type=int, default=10, help="number of output classes")
    parser.add_argument("--model", type=str, default='gin', choices=['gcn', 'gin', 'gat', 'sage'], help="type of model")
    parser.add_argument("--training", type=str, default='training', choices=['training', 'inference'], help="training or inference")
    parser.add_argument("--aggregator_type", type=str, default='mean', choices=['mean', 'gcn', 'pool'], help="aggregator_type")
    parser.add_argument("--num_layers", type=int, default=1,
                            help="number of gat layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_outheads", type=int, default=1,
                        help="number of output attention heads")                        
    parser.add_argument("--torch_profile", type=str2bool, default=False,  help="torch_profile")
    parser.add_argument("--in_drop", type=float, default=.0,
                            help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.0,
                            help="attention dropout")

    parser.add_argument('--negative-slope', type=float, default=0.2,
                            help="the negative slope of leaky relu")

    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument('--early_stop', action='store_true', default=False,
                            help="indicates whether to use early stop or not")
    args = parser.parse_args()
    print(args)

    if args.model == 'gcn':
        from models import GCN
    elif args.model == 'gat':
        from models import GAT
    elif args.model == 'sage':
        from models import GraphSAGE
    elif args.model == 'gin':
        from models import GIN

    dgltester = DGLTester(dataDir=args.dataDir,
                    model=args.model,
                    dataset=args.dataset,
                    dim=args.dim,
                    classes=args.classes,
                    gpu=args.gpu,
                    training=args.training,
                    torch_profile=args.torch_profile,
                    n_epochs=args.n_epochs,
                    hidden=args.hidden,
                    aggregator_type=args.aggregator_type,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    num_outheads=args.num_outheads,
                    in_drop=args.in_drop,
                    attn_drop=args.attn_drop,
                    negative_slope=args.negative_slope,
                    residual=args.residual

                    )
    dgltester.main()

