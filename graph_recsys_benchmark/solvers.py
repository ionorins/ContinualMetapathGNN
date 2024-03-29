import os
import random as rd
from secrets import token_hex
import numpy as np
import torch
import time
import pandas as pd
import tqdm
from torch.utils.data import DataLoader

from graph_recsys_benchmark.utils import *


class BaseSolver(object):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        self.model_class = model_class

        self.dataset_args = dataset_args
        self.model_args = model_args
        self.train_args = train_args

    def generate_candidates(self, dataset, u_nid):
        """
        Return the recommendation candidates to the algorithms to rank
        :param dataset: graph_recsys_benchmark.dataset.Dataset object
        :param u_nid: user node ids
        :return:
        """
        pos_i_nids = dataset.test_pos_unid_inid_map[u_nid]
        neg_i_nids = list(np.random.choice(dataset.neg_unid_inid_map[u_nid], size=(
            self.train_args['num_neg_candidates'],)))

        return pos_i_nids, neg_i_nids

    def metrics(
            self,
            run,
            epoch,
            model,
            dataset
    ):
        """
        Generate the positive and negative candidates for the recsys evaluation
        :param run:
        :param epoch:
        :param model:
        :param dataset:
        :return: a tuple (pos_i_nids, neg_i_nids), two entries should be both list
        """
        HRs, NDCGs, AUC, eval_losses = np.zeros((0, 16)), np.zeros(
            (0, 16)), np.zeros((0, 1)), np.zeros((0, 1))
        accs = np.zeros((0, 1))

        test_pos_unid_inid_map, neg_unid_inid_map = \
            dataset.test_pos_unid_inid_map, dataset.neg_unid_inid_map

        u_nids = list(test_pos_unid_inid_map.keys())
        test_bar = tqdm.tqdm(u_nids, total=len(u_nids))
        for u_idx, u_nid in enumerate(test_bar):
            pos_i_nids, neg_i_nids = self.generate_candidates(
                dataset, u_nid
            )
            if len(pos_i_nids) == 0 or len(neg_i_nids) == 0:
                # print("No pos or neg samples found in evaluation!")
                continue

            pos_i_nid_df = pd.DataFrame(
                {'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
            neg_i_nid_df = pd.DataFrame(
                {'u_nid': [u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
            pos_neg_pair_t = torch.from_numpy(
                pd.merge(pos_i_nid_df, neg_i_nid_df,
                         how='inner', on='u_nid').to_numpy()
            ).to(self.train_args['device'])

            if self.model_args['model_type'] == 'MF':
                pos_neg_pair_t[:, 0] -= dataset.e2nid_dict['uid'][0]
                pos_neg_pair_t[:, 1:] -= dataset.e2nid_dict['iid'][0]
            y = torch.tensor([dataset.edge_values.get((user.item(), item.item()), 0) for user, item in pos_neg_pair_t[:, :2]]).to(self.train_args['device'])
            loss = model.loss((pos_neg_pair_t, y)).detach().cpu().item()

            pos_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(pos_i_nids))])).to(
                self.train_args['device'])
            pos_i_nids_t = torch.from_numpy(
                np.array(pos_i_nids)).to(self.train_args['device'])
            neg_u_nids_t = torch.from_numpy(np.array([u_nid for _ in range(len(neg_i_nids))])).to(
                self.train_args['device'])
            neg_i_nids_t = torch.from_numpy(
                np.array(neg_i_nids)).to(self.train_args['device'])
            if self.model_args['model_type'] == 'MF':
                pos_u_nids_t -= dataset.e2nid_dict['uid'][0]
                neg_u_nids_t -= dataset.e2nid_dict['uid'][0]
                pos_i_nids_t -= dataset.e2nid_dict['iid'][0]
                neg_i_nids_t -= dataset.e2nid_dict['iid'][0]
            pos_pred = model.predict(pos_u_nids_t, pos_i_nids_t).reshape(-1)
            neg_pred = model.predict(neg_u_nids_t, neg_i_nids_t).reshape(-1)



            _, indices = torch.sort(
                torch.cat([pos_pred, neg_pred]), descending=True)
            hit_vec = (indices < len(pos_i_nids)).cpu().detach().numpy()
            pos_pred = pos_pred.cpu().detach().numpy()
            neg_pred = neg_pred.cpu().detach().numpy()

            # calculate accuracy
            ground_truth = np.array([dataset.edge_values[(u_nid, i_nid)] >= 0.5 for i_nid in pos_i_nids])
            ground_truth = np.concatenate([ground_truth, np.zeros(len(neg_i_nids))])
            all_pred = np.concatenate([pos_pred, neg_pred])
            acc = np.equal(ground_truth, (all_pred >= 0.5)).mean()
            # acc = torch.cat([pos_pred >= 0.5, neg_pred < 0.5]).float().mean().item()
            accs = np.vstack([accs, acc])

            
            # apply sigmoid to predictions
            # sigmoid = lambda x: 1 / (1 + np.exp(-x))
            # pos_pred = sigmoid(pos_pred)
            # calculate r2 score
            # r2 = r2_score(ground_truth, pos_pred)
            # compute accuracy
            # acc = (ground_truth == np.round(pos_pred, 1)).mean()
            # accs = np.vstack([accs, acc])

            HRs = np.vstack([HRs, hit(hit_vec)])
            NDCGs = np.vstack([NDCGs, ndcg(hit_vec)])
            AUC = np.vstack([AUC, auc(pos_pred, neg_pred)])
            eval_losses = np.vstack([eval_losses, loss])
            test_bar.set_description(
                'Run {}, epoch: {}, acc: {:.4f} HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, eval loss: {:.4f} '.format(
                    run, epoch, accs.mean(axis=0)[0], HRs.mean(axis=0)[0], HRs.mean(
                        axis=0)[5], HRs.mean(axis=0)[10], HRs.mean(axis=0)[15],
                    NDCGs.mean(axis=0)[0], NDCGs.mean(axis=0)[5], NDCGs.mean(
                        axis=0)[10], NDCGs.mean(axis=0)[15],
                    AUC.mean(axis=0)[0], eval_losses.mean(axis=0)[0])
            )
        return np.mean(HRs, axis=0), np.mean(NDCGs, axis=0), np.mean(AUC, axis=0), np.mean(eval_losses, axis=0), np.mean(accs, axis=0)

    def run(self):
        # save model weights under random name
        model_filename = token_hex(16)

        self.dataset_args['token'] = model_filename
        
        # if end timeframe is not specified, set it to be the last one
        if self.dataset_args['end_timeframe'] is None:
            self.dataset_args['end_timeframe'] = self.dataset_args['num_timeframes']

        # iterate through timeframes
        for i in range(self.dataset_args['start_timeframe'], self.dataset_args['end_timeframe']):
            print(f'Timeframe {i}')

            # delete checkpoints so that the program does not load wrong weights
            import shutil

            shutil.rmtree('checkpoint/loggers', ignore_errors=True)
            shutil.rmtree('checkpoint/weights', ignore_errors=True)
            shutil.rmtree(
                'checkpoint/data/Movielenslatest-small/processed' + model_filename, ignore_errors=True)

            global_logger_path = self.train_args['logger_folder']
            if not os.path.exists(global_logger_path):
                os.makedirs(global_logger_path, exist_ok=True)
            global_logger_file_path = os.path.join(
                global_logger_path, 'global_logger.pkl')
            HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
                load_global_logger(global_logger_file_path)

            # Create the dataset
            self.dataset_args['timeframe'] = i

            dataset = load_dataset(self.dataset_args)

            logger_file_path = os.path.join(
                global_logger_path, 'logger_file.txt')
            with open(logger_file_path, 'a') as logger_file:
                start_run = last_run + 1
                if start_run <= self.train_args['runs']:
                    for run in range(start_run, self.train_args['runs'] + 1):
                        # Fix the random seed
                        seed = 2019 + run
                        rd.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)

                        # Create model and optimizer
                        if self.model_args['model_type'] == 'Graph':
                            if self.model_args['if_use_features']:
                                self.model_args['emb_dim'] = dataset.data.x.shape[1]
                            self.model_args['num_nodes'] = dataset.num_nodes
                            self.model_args['dataset'] = dataset
                        elif self.model_args['model_type'] == 'MF':
                            self.model_args['num_users'] = dataset.num_uids
                            self.model_args['num_items'] = dataset.num_iids

                        if i == self.dataset_args['start_timeframe'] or dataset.continual_aspect in ['single', 'retrained']:
                            model = self.model_class(**self.model_args)
                            model = model.to(self.train_args['device'])
                            last_emb = None
                            crt_emb = None
                            if self.train_args['train_between_emb_diff']:
                                crt_emb = model.forward()
                        else:
                            model = torch.load('pths/' + model_filename + '.pth')

                            if self.train_args['train_between_emb_diff']:
                                last_emb = crt_emb
                            else:
                                last_emb = model.forward()

                            model.update_graph_input(dataset)
                            crt_emb = model.forward()

                        opt_class = get_opt_class(self.train_args['opt'])
                        optimizer = opt_class(
                            params=model.parameters(),
                            lr=self.train_args['lr'],
                            weight_decay=self.train_args['weight_decay']
                        )

                        # Load models
                        weights_path = os.path.join(
                            self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                        if not os.path.exists(weights_path):
                            os.makedirs(weights_path, exist_ok=True)
                        weights_file = os.path.join(weights_path, 'latest.pkl')
                        model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                               self.train_args['device'])
                        HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np = \
                            rec_metrics

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        start_epoch = last_epoch + 1
                        if start_epoch == 1 and self.train_args['init_eval']:
                            new_model = None
                            if dataset.continual_aspect in ['single', 'retrained']:
                                new_model = model
                                if i > self.dataset_args['start_timeframe']:
                                    model = torch.load(model_filename + '.pth')
                            model.eval()
                            with torch.no_grad():
                                HRs_before_np, NDCGs_before_np, AUC_before_np, cf_eval_loss_before_np, accs_before_np = \
                                    self.metrics(run, 0, model, dataset)

                            if dataset.continual_aspect in ['single', 'retrained']:
                                model = new_model

                            print(
                                'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '
                                'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                    HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                                    NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                                    AUC_before_np[0], cf_eval_loss_before_np[0]
                                )
                            )

                            # save metrics
                            if dataset.skip_timeframe and not dataset.future_testing:
                                # f = open(f'HRs/{dataset.num_timeframes}{dataset.continual_aspect}{self.train_args["out_filename"]}.csv', 'a')
                                # f.write(f'{i},{str(HRs_before_np[5])},{str(NDCGs_before_np[5])}\n')
                                # f.close()
                                break

                            if dataset.future_testing and i > self.dataset_args['start_timeframe']:
                                # f = open(f'HRs/{dataset.num_timeframes}{dataset.continual_aspect}{self.train_args["out_filename"]}future.csv', 'a')
                                # f.write(f'{i},{str(HRs_before_np[5])},{str(NDCGs_before_np[5])}\n')
                                # f.close()

                                if dataset.continual_aspect == 'pretrained':
                                    break

                            logger_file.write(
                                'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '
                                'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                                    HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                                    NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                                    AUC_before_np[0], cf_eval_loss_before_np[0]
                                )
                            )
                            instantwrite(logger_file)
                            clearcache()

                        t_start = time.perf_counter()

                        if start_epoch <= self.train_args['epochs']:
                            # Start training model
                            # if first timeframe

                            no_epochs = self.train_args['epochs'] + 1
                            if i < self.train_args['theta']:
                                no_epochs = int(self.train_args['theta'] * self.train_args['epochs'] / (i + 1) + 1)

                            for epoch in range(start_epoch, no_epochs):
                                loss_per_batch = []
                                model.train()
                                dataset.cf_negative_sampling(
                                    last_emb, 
                                    crt_emb, 
                                    self.train_args['theta'], 
                                    epoch
                                )

                                print(
                                    f'len(dataset.train_data)={len(dataset.train_data)}')

                                train_dataloader = DataLoader(
                                    dataset,
                                    shuffle=True,
                                    batch_size=self.train_args['batch_size'],
                                    num_workers=self.train_args['num_workers']
                                )
                                train_bar = tqdm.tqdm(
                                    train_dataloader, total=len(train_dataloader))

                                for _, batch in enumerate(train_bar):
                                    if self.model_args['model_type'] == 'MF':
                                        if self.model_args['loss_type'] == 'BCE':
                                            batch[:, 0] -= dataset.e2nid_dict['uid'][0]
                                            batch[:, 1] -= dataset.e2nid_dict['iid'][0]
                                        elif self.model_args['loss_type'] == 'BPR':
                                            batch[:, 0] -= dataset.e2nid_dict['uid'][0]
                                            batch[:, 1:] -= dataset.e2nid_dict['iid'][0]
                                        
                                    batch = batch[0].to(self.train_args['device']), batch[1].to(self.train_args['device'])

                                    optimizer.zero_grad()
                                    loss = model.loss(batch)
                                    loss.backward()
                                    optimizer.step()

                                    loss_per_batch.append(
                                        loss.detach().cpu().item())
                                    train_loss = np.mean(loss_per_batch)
                                    train_bar.set_description(
                                        'Run: {}, epoch: {}, train loss: {:.4f}'.format(
                                            run, epoch, train_loss)
                                    )

                                if model.__class__.__name__[:3] == 'PEA' and self.train_args['metapath_test']:
                                    if (self.dataset_args['dataset'] == 'Movielens' and epoch == 30) or (self.dataset_args['dataset'] == 'Yelp' and epoch == 20):
                                        for metapath_idx in range(len(self.model_args['meta_path_steps'])):
                                            model.eval(metapath_idx)
                                            HRs, NDCGs, AUC, eval_loss, accs = self.metrics(
                                                run, epoch, model, dataset)
                                            print(
                                                'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                                    run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[
                                                        15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                                    AUC[0], train_loss, eval_loss[0]
                                                )
                                            )
                                            logger_file.write(
                                                'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                                    run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[
                                                        15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                                    AUC[0], train_loss, eval_loss[0]
                                                )
                                            )

                                model.eval()

                                # save embeddings
                                # df = pd.DataFrame(model.forward().cpu().detach().numpy())
                                # df.to_csv(f'timeframe{i}emb.csv')

                                # save attention values
                                # df = pd.DataFrame(model.att[0].cpu().detach().numpy())
                                # df.to_csv(f'timeframe{i}att.csv')
                                
                                with torch.no_grad():
                                    HRs, NDCGs, AUC, eval_loss, accs = self.metrics(
                                        run, epoch, model, dataset)

                                # Sumarize the epoch
                                HRs_per_epoch_np = np.vstack(
                                    [HRs_per_epoch_np, HRs])
                                NDCGs_per_epoch_np = np.vstack(
                                    [NDCGs_per_epoch_np, NDCGs])
                                AUC_per_epoch_np = np.vstack(
                                    [AUC_per_epoch_np, AUC])
                                train_loss_per_epoch_np = np.vstack(
                                    [train_loss_per_epoch_np, np.array([train_loss])])
                                eval_loss_per_epoch_np = np.vstack(
                                    [eval_loss_per_epoch_np, np.array([eval_loss])])

                                if epoch in self.train_args['save_epochs']:
                                    weightpath = os.path.join(
                                        weights_path, '{}.pkl'.format(epoch))
                                    save_model(
                                        weightpath,
                                        model, optimizer, epoch,
                                        rec_metrics=(
                                            HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                    )
                                if epoch > self.train_args['save_every_epoch']:
                                    weightpath = os.path.join(
                                        weights_path, 'latest.pkl')
                                    save_model(
                                        weightpath,
                                        model, optimizer, epoch,
                                        rec_metrics=(
                                            HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                    )
                                print(
                                    'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                    'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                        run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                        AUC[0], train_loss, eval_loss[0]
                                    )
                                )
                                logger_file.write(
                                    'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                    'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                        run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                        AUC[0], train_loss, eval_loss[0]
                                    )
                                )
                                instantwrite(logger_file)
                                clearcache()
                            f = open(f'HRs/{self.train_args["out_filename"]}.csv', 'a')
                            f.write(f'{i},{str(accs[0])},{str(HRs[5])},{str(NDCGs[5])}\n')
                            f.close()

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        t_end = time.perf_counter()

                        # save ewc parameters and model
                        # model.register_ewc_params(dataset.train_data)
                        # print('REGISTERED EWC PARAMS')
                        torch.save(model, 'pths/' + model_filename + '.pth')

                        HRs_per_run_np = np.vstack(
                            [HRs_per_run_np, np.max(HRs_per_epoch_np, axis=0)])
                        NDCGs_per_run_np = np.vstack(
                            [NDCGs_per_run_np, np.max(NDCGs_per_epoch_np, axis=0)])
                        AUC_per_run_np = np.vstack(
                            [AUC_per_run_np, np.max(AUC_per_epoch_np, axis=0)])
                        train_loss_per_run_np = np.vstack(
                            [train_loss_per_run_np, np.mean(train_loss_per_epoch_np, axis=0)])
                        eval_loss_per_run_np = np.vstack(
                            [eval_loss_per_run_np, np.mean(eval_loss_per_epoch_np, axis=0)])

                        save_global_logger(
                            global_logger_file_path,
                            HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                            train_loss_per_run_np, eval_loss_per_run_np
                        )
                        print(
                            'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                                run, t_end -
                                t_start, np.max(HRs_per_epoch_np, axis=0)[
                                    0], np.max(HRs_per_epoch_np, axis=0)[5],
                                np.max(HRs_per_epoch_np, axis=0)[10], np.max(
                                    HRs_per_epoch_np, axis=0)[15],
                                np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[
                                        5], np.max(NDCGs_per_epoch_np, axis=0)[10],
                                np.max(NDCGs_per_epoch_np, axis=0)[
                                        15],  np.max(AUC_per_epoch_np, axis=0)[0],
                                train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                        )
                        logger_file.write(
                            'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                                run, t_end -
                                t_start, np.max(HRs_per_epoch_np, axis=0)[
                                    0], np.max(HRs_per_epoch_np, axis=0)[5],
                                np.max(HRs_per_epoch_np, axis=0)[10], np.max(
                                    HRs_per_epoch_np, axis=0)[15],
                                np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(
                                    NDCGs_per_epoch_np, axis=0)[5],
                                np.max(NDCGs_per_epoch_np, axis=0)[10], np.max(
                                    NDCGs_per_epoch_np, axis=0)[15],
                                np.max(AUC_per_epoch_np, axis=0)[0],
                                train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                        )
                        instantwrite(logger_file)

                        del model, optimizer, loss, loss_per_batch, rec_metrics, train_dataloader
                        clearcache()

                if self.dataset_args['model'][:3] == 'PEA' and self.train_args['metapath_test']:
                    run = 1
                    if self.dataset_args['dataset'] == 'Movielens':
                        epoch = 30
                    if self.dataset_args['dataset'] == 'Yelp':
                        epoch = 20
                    seed = 2019 + run
                    rd.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    # Create model and optimizer
                    if self.model_args['model_type'] == 'Graph':
                        if self.model_args['if_use_features']:
                            self.model_args['emb_dim'] = dataset.data.x.shape[1]
                        self.model_args['num_nodes'] = dataset.num_nodes
                        self.model_args['dataset'] = dataset
                    elif self.model_args['model_type'] == 'MF':
                        self.model_args['num_users'] = dataset.num_uids
                        self.model_args['num_items'] = dataset.num_iids

                    model = self.model_class(
                        **self.model_args).to(self.train_args['device'])

                    opt_class = get_opt_class(self.train_args['opt'])
                    optimizer = opt_class(
                        params=model.parameters(),
                        lr=self.train_args['lr'],
                        weight_decay=self.train_args['weight_decay']
                    )

                    # Load models
                    weights_path = os.path.join(
                        self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)
                    weights_file = os.path.join(weights_path, 'latest.pkl')
                    model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                           self.train_args['device'])
                    for metapath_idx in range(len(self.model_args['meta_path_steps'])):
                        model.eval(metapath_idx)
                        HRs, NDCGs, AUC, eval_loss, accs = self.metrics(
                            run, epoch, model, dataset)
                        print(
                            'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            '\n'.format(
                                run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                NDCGs[15],
                                AUC[0]
                            )
                        )
                        logger_file.write(
                            'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                            'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                            '\n'.format(
                                run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                NDCGs[15],
                                AUC[0]
                            )
                        )
                        instantwrite(logger_file)

                if dataset.skip_timeframe:
                    continue

                if not dataset.future_testing:
                    print(
                        'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(
                                axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                            HRs_per_run_np.mean(
                                axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                            NDCGs_per_run_np.mean(
                                axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                            NDCGs_per_run_np.mean(
                                axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                            train_loss_per_run_np.mean(
                                axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                        )
                    )
                    logger_file.write(
                        'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(
                                axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                            HRs_per_run_np.mean(
                                axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                            NDCGs_per_run_np.mean(
                                axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                            NDCGs_per_run_np.mean(
                                axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                            train_loss_per_run_np.mean(
                                axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                        )
                    )
                    instantwrite(logger_file)

                    # f = open(f'HRs/{dataset.num_timeframes}{dataset.continual_aspect}{self.train_args["out_filename"]}.csv', 'a')
                    # f.write(f'{i},{str(HRs_per_run_np[-1][5])},{str(NDCGs_per_run_np[-1][5])}\n')
                    # f.close()
