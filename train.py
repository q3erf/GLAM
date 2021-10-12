import torch
import time
from data.data_loader_multigraph import GMDataset, get_dataloader
from utils.config import cfg
from module.model import Net
from utils.utils import update_params_from_cmdline
from eval import eval_model
from utils.evaluation_metric import matching_accuracy_from_lists, f1_score, get_pos_neg_from_lists


def train_model(model, optimizer, dataloader, criterion,num_epochs):
    since = time.time()
    print("Start training...")
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0
        for inputs in dataloader['train']:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            iter_num += 1
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # print(sum(n_points_gt_list[0]))
                s_pred_list = model(data_list, points_gt_list, n_points_gt_list)

                s_pred_list = s_pred_list.view(s_pred_list.size(0), -1).squeeze()
                s_pred_list = torch.stack([s_pred_list, 1-s_pred_list], dim=2)
                perm_mat_list = perm_mat_list[0].view(perm_mat_list[0].size(0), -1).squeeze().long()

                loss = sum([criterion(s_pred, perm_mat) for s_pred, perm_mat in zip(s_pred_list, perm_mat_list)])
                loss /= len(s_pred_list)

                loss.backward()
                optimizer.step()
                # tp, fp, fn = get_pos_neg_from_lists(s_pred_list, perm_mat_list)
                # f1 = f1_score(tp, fp, fn)
                # acc, _, __ = matching_accuracy_from_lists(s_pred_list, perm_mat_list)

                bs = perm_mat_list[0].size(0)
                running_loss += loss.item() * bs  # multiply with batch size
                epoch_loss += loss.item() * bs

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * bs / (time.time() - running_since)
                    loss_avg = running_loss / cfg.STATISTIC_STEP / bs
                    print(
                        "Epoch {:<4} Iter {:<4} {:>4.2f}sample/s Loss={:<8.4f} ".format(
                            epoch, iter_num, running_speed, loss_avg,
                        )
                    )

                    running_loss = 0.0
                    running_since = time.time()

            epoch_loss = epoch_loss / dataset_size

        print(f"Over whole epoch {epoch:<4} -------- Loss: {epoch_loss:.4f} ")
        # Eval in each epoch
        accs, f1_scores = eval_model(model, dataloader["test"])
        acc_dict = {
            "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
        }
        f1_dict = {
            "f1_{}".format(cls): single_f1_score
            for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
        }
        acc_dict.update(f1_dict)
        acc_dict["matching_accuracy"] = torch.mean(accs)
        acc_dict["f1_score"] = torch.mean(f1_scores)


if __name__ == '__main__':

    cfg = update_params_from_cmdline(default_params=cfg)
    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test")) for x in ("train", "test")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.cuda()

    backbone_params = list(model.node_layers.parameters()) + list(model.edge_layers.parameters())
    backbone_params += list(model.final_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = torch.optim.Adam(opt_params)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([5,1]).cuda())

    train_model(model, optimizer,dataloader, criterion, 10)


