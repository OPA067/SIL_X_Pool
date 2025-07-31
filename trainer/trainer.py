import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
import torch.nn.functional as F
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, np_softmax

class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 test_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best = -1.0

        self.test_batch_size = config.batch_size

        self.split_batch = config.split

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        start_time = time.time()

        if epoch == 1:
            _, Rsum = self._valid_epoch_step(epoch, 0, num_steps - 1)
            msg = (" Zero-Shot of Current Text-Video R@sum is {}".format(Rsum))
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)
            self._save_checkpoint(epoch - 1, save_best=False)

        for batch_idx, data in enumerate(self.train_data_loader):
            data['cap'] = self.tokenizer(data['cap'], return_tensors='pt', padding=True, truncation=True)
            data['cap'] = {key: val.to(self.device) for key, val in data['cap'].items()}

            for i in range(self.config.num_captions):
                seq = data['seq_cap'][i]
                seq = self.tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
                data['seq_cap'][i] = {key: val.to(self.device) for key, val in seq.items()}

            data['video'] = data['video'].to(self.device)

            sims_cf, sims_sf, sims_sc, loss_kl = self.model(data, is_train=True)
            sims_cf_loss = self.loss(sims_cf, self.model.clip.logit_scale)
            sims_sf_loss = self.loss(sims_sf, self.model.clip.logit_scale)
            sims_sc_loss = self.loss(sims_sc, self.model.clip.logit_scale)
            loss_all = sims_cf_loss + sims_sf_loss + sims_sc_loss + loss_kl

            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            cost_time = time.time() - start_time
            start_time = time.time()

            eta_time = (len(self.train_data_loader) * self.config.num_epochs - batch_idx * epoch) * cost_time
            eta_time = f"{int(eta_time // 3600):02}:{int((eta_time % 3600) // 60):02}:{int(eta_time % 60):02}"

            if batch_idx % self.log_step == 0:
                msg = (
                'Train epoch: {} dl:{}/{} total_loss:{:.10f}, eta_time:{}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss_all.detach().item(),
                    eta_time
                ))
                gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            if batch_idx in eval_steps:

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

                else:
                    test_res, Rsum = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                    self.model.train()

                    if Rsum > self.best:
                        self.best = Rsum
                        self._save_checkpoint(epoch, save_best=True)

                    msg = (" Current Best Text-Video R@sum is {}".format(self.best))
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
                    gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

        res = {
            'loss_train': total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        s_embed_arr, c_embed_arr, f_embed_arr = [], [], []

        start_selection_time = time.time()

        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.test_data_loader)):
                data['cap'] = self.tokenizer(data['cap'], return_tensors='pt', padding=True, truncation=True)
                data['cap'] = {key: val.to(self.device) for key, val in data['cap'].items()}

                for i in range(self.config.num_captions):
                    seq = data['seq_cap'][i]
                    seq = self.tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
                    data['seq_cap'][i] = {key: val.to(self.device) for key, val in seq.items()}

                data['video'] = data['video'].to(self.device)

                s_feat, c_feat, f_feat = self.model(data, is_train=False)
                s_embed_arr.append(s_feat)
                c_embed_arr.append(c_feat)
                f_embed_arr.append(f_feat)

            s_embed = torch.cat(s_embed_arr, dim=0)
            c_embed = torch.cat(c_embed_arr, dim=0)
            f_embed = torch.cat(f_embed_arr, dim=0)

            batch_s_feat = torch.split(s_embed, self.split_batch)
            batch_c_feat = torch.split(c_embed, self.split_batch)
            batch_f_feat = torch.split(f_embed, self.split_batch)
            sim_matrix = []
            for idx1, s_feat in tqdm(enumerate(batch_s_feat)):
                each_row = []
                for idx2, (c_feat, f_feat) in enumerate(zip(batch_c_feat, batch_f_feat)):
                    logits = self.model.get_similarity_logits(s_feat, c_feat, f_feat)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

            del batch_s_feat, batch_c_feat, batch_f_feat
            gc.collect()

            if self.config.DSL:
                sims_t2v = sim_matrix * np_softmax(sim_matrix * 100, axis=0)
            else:
                sims_t2v = sim_matrix
            metrics = self.metrics
            res = metrics(sims_t2v)
            Rsum = res['Rsum']
            msg = (f"--text-video--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"R@sum: {res['Rsum']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MnR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            ''' Here we conduct video-text retrieval (.T)'''
            sim_matrix = sim_matrix.T
            if self.config.DSL:
                sims_v2t = sim_matrix * np_softmax(sim_matrix * 100, axis=0)
            else:
                sims_v2t = sim_matrix
            res = metrics(sims_v2t)
            msg = (f"--video-text--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"Rsum: {res['Rsum']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MnR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            end_selection_time = time.time()

            msg = (
                f'To compute all video-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time}')
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            return res, Rsum
