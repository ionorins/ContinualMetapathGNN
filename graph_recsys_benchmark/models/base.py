import torch
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from torch_geometric.nn.inits import glorot, zeros
from torch import autograd


class BaseRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError


class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()

        self.ewc_type = kwargs['ewc_type']
        self.ewc_lambda = kwargs['ewc_lambda']

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def real_loss(self, pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        if self.entity_aware and self.training:
            pos_item_entity, neg_item_entity = pos_neg_pair_t[:,
                                                              3], pos_neg_pair_t[:, 4]
            pos_user_entity, neg_user_entity = pos_neg_pair_t[:,
                                                              6], pos_neg_pair_t[:, 7]
            item_entity_mask, user_entity_mask = pos_neg_pair_t[:,
                                                                5], pos_neg_pair_t[:, 8]

            # l2 norm
            x = self.x
            item_pos_reg = (x[pos_neg_pair_t[:, 1]] - x[pos_item_entity]) * (
                x[pos_neg_pair_t[:, 1]] - x[pos_item_entity])
            item_neg_reg = (x[pos_neg_pair_t[:, 1]] - x[neg_item_entity]) * (
                x[pos_neg_pair_t[:, 1]] - x[neg_item_entity])
            item_pos_reg = item_pos_reg.sum(dim=-1)
            item_neg_reg = item_neg_reg.sum(dim=-1)

            user_pos_reg = (x[pos_neg_pair_t[:, 0]] - x[pos_user_entity]) * (
                x[pos_neg_pair_t[:, 0]] - x[pos_user_entity])
            user_neg_reg = (x[pos_neg_pair_t[:, 0]] - x[neg_user_entity]) * (
                x[pos_neg_pair_t[:, 0]] - x[neg_user_entity])
            user_pos_reg = user_pos_reg.sum(dim=-1)
            user_neg_reg = user_neg_reg.sum(dim=-1)

            item_reg_los = -((item_pos_reg - item_neg_reg) *
                             item_entity_mask).sigmoid().log().sum()
            user_reg_los = -((user_pos_reg - user_neg_reg) *
                             user_entity_mask).sigmoid().log().sum()
            reg_los = item_reg_los + user_reg_los

            # two parts of loss
            loss = cf_loss + self.entity_aware_coff * reg_los
        else:
            loss = cf_loss

        return loss

    def update_graph_input(self, dataset):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError

    def eval(self, metapath_idx=None):
        super(GraphRecsysModel, self).eval()
        if self.__class__.__name__ not in ['KGATRecsysModel', 'KGCNRecsysModel']:
            if self.__class__.__name__[:3] == 'PEA':
                with torch.no_grad():
                    self.cached_repr = self.forward(metapath_idx)
            else:
                with torch.no_grad():
                    self.cached_repr = self.forward()

    def _update_mean_params(self):
        for param_name, param in self.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name +
                                 '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, pos_neg_pair_t):
        log_likelihood = self.real_loss(pos_neg_pair_t)
        grad_log_liklihood = autograd.grad(log_likelihood, self.parameters(), allow_unused=True)
        _buff_param_names = [param[0].replace(
            '.', '__') for param in self.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            if param is None:
                continue
            # print('_buff_param_name:', _buff_param_name)
            self.register_buffer(_buff_param_name +
                                 '_estimated_fisher', param.data.clone() ** 2)

    def _save_fisher_params(self):
        for param_name, param in self.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(
                self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = np.array(
                getattr(self, '{}_estimated_fisher'.format(_buff_param_name)))
            np.savetxt('estimated_mean', estimated_mean)
            np.savetxt('estimated_fisher', estimated_fisher)
            print(np.mean(estimated_fisher), np.max(
                estimated_fisher), np.min(estimated_fisher))
            break

    def register_ewc_params(self, pos_neg_pair_t):
        self._update_fisher_params(pos_neg_pair_t)
        self._update_mean_params()

    def _compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.named_parameters():
            try:
                # print("param_name", param_name)
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(
                    self, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(
                    self, '{}_estimated_fisher'.format(_buff_param_name))
                if self.ewc_type == 'l2':
                    losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
                else:
                    losses.append(
                        (estimated_fisher * (param - estimated_mean) ** 2).sum())
                # print('_buff_param_name:', _buff_param_name)
            except:
                pass
        return 1 * (self.ewc_lambda / 2) * sum(losses)

    def loss(self, pos_neg_pair_t):
        loss1 = self.real_loss(pos_neg_pair_t)
        loss2 = 0
        try:
            loss2 = self._compute_consolidation_loss()
        except Exception as e:
            print(e)
        # print(f'loss1: {loss1} loss2: {loss2}')

        loss = loss1 + loss2
        return loss

class MFRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MFRecsysModel, self).__init__()
        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        loss_func = torch.nn.BCEWithLogitsLoss()
        if self.training:
            pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
            label = pos_neg_pair_t[:, -1].float()
        else:
            pos_pred = self.predict(
                pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])[:1]
            neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([torch.ones_like(pos_pred),
                               torch.zeros_like(neg_pred)]).float()

        loss = loss_func(pred, label)
        return loss

    def predict(self, unids, inids):
        return self.forward(unids, inids)


class PEABaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        assert len(edge_index_list) == self.num_steps

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gnn_layers[-1](x, edge_index_list[-1])
        return x


class PEABaseRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEABaseRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.channel_aggr = kwargs['channel_aggr']

        # Create node embedding
        if not self.if_use_features:
            self.x = Parameter(torch.eye(kwargs['dataset']['num_nodes']))
            self.x.requires_grad = False
            kwargs['emb_dim'] = kwargs['dataset']['num_nodes']
        else:
            raise NotImplementedError('Feature not implemented!')

        # Create graphs
        meta_path_edge_index_list = self.update_graph_input(kwargs['dataset'])
        assert len(meta_path_edge_index_list) == len(kwargs['meta_path_steps'])
        self.meta_path_edge_index_list = meta_path_edge_index_list

        # Create channels
        self.pea_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.pea_channels.append(kwargs_cpy['channel_class'](**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = Parameter(torch.Tensor(
                1, len(kwargs['meta_path_steps']), kwargs['repr_dim']))

        if self.channel_aggr == 'cat':
            self.fc1 = torch.nn.Linear(
                2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            self.fc1 = torch.nn.Linear(
                2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        for module in self.pea_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att)

    def forward(self, metapath_idx=None):
        x = self.x
        x = [module(x, self.meta_path_edge_index_list[idx]).unsqueeze(1)
             for idx, module in enumerate(self.pea_channels)]
        if metapath_idx is not None:
            x[metapath_idx] = torch.zeros_like(x[metapath_idx])
        x = torch.cat(x, dim=1)
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1),
                             dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
