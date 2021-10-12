import torch
import time
from utils.config import cfg
from utils.hungarian import hungarian, hungarian_from_single
from utils.evaluation_metric import matching_accuracy_from_lists, f1_score, get_pos_neg_from_lists

def eval_model(model, dataloader):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    ds.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):

        running_since = time.time()
        iter_num = 0

        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)
        for k, inputs in enumerate(dataloader):
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            batch_num = data_list[0].size(0)

            iter_num = iter_num + 1

            visualize = k == 0 and cfg.visualize
            visualization_params = {**cfg.visualization_params, **dict(string_info=cls, true_matchings=perm_mat_list)}
            with torch.set_grad_enabled(False):
                s_pred_list = model(data_list, points_gt_list, n_points_gt_list)
            s_pred_list = [hungarian_from_single(s_pred[0:point_n, 0:point_n]) for s_pred,point_n in zip(s_pred_list, n_points_gt_list[0])]
            perm_mat_list = [perm_mat[0:point_n, 0:point_n] for perm_mat, point_n in zip(perm_mat_list[0], n_points_gt_list[0])]

            _, _acc_match_num, _acc_total_num = matching_accuracy_from_lists(s_pred_list, perm_mat_list)
            _tp, _fp, _fn = get_pos_neg_from_lists(s_pred_list, perm_mat_list)

            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            tp += _tp
            fp += _fp
            fn += _fn

            if iter_num % cfg.STATISTIC_STEP == 0 :
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        f1_scores[i] = f1_score(tp, fp, fn)

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    return accs, f1_scores