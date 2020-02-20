import apex
from apex import amp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_toolbelt.inference import tta as pytta

from cvcore.data import cutmix_data, mixup_data, mixup_criterion
from cvcore.modeling.loss import binary_iou_metric, binary_dice_metric
from cvcore.modeling.loss import binary_dice_loss, binary_iou_loss
from cvcore.utils import AverageMeter, save_checkpoint


def valid_model(_print, cfg, model, valid_loader,
                optimizer, epoch, cycle=None,
                best_metric=None, checkpoint=False):
    if cfg.INFER.TTA:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    # switch to evaluate mode
    model.eval()

    # valid_iou = []
    tbar = tqdm(valid_loader)
    outputs = []
    masks = []


    with torch.no_grad():
        for i, (image, mask) in enumerate(tbar):
            image = image.cuda()
            mask = mask.cuda()
            output = model(image)
            outputs.append(output)
            masks.append(mask)
            # batch_iou = binary_dice_metric(output, mask).cpu()
            # valid_iou.append(batch_iou)
        outputs = torch.cat(outputs,0)
        masks = torch.cat(masks,0)
        valid_iou = binary_dice_metric(outputs, masks).cpu().mean(0).numpy()

    # record IoU over foreground classes and background
    # TODO: compute IoU for background
    # valid_iou = torch.cat(valid_iou, 0).mean(0).numpy()
    final_score = np.average(valid_iou)
    log_info = "Mean 2IoInU: %.4f - mask0: %.4f - mask1: %.4f"
    _print(log_info % (final_score, valid_iou[0], valid_iou[1]))
    # checkpoint
    if checkpoint:
        is_best = final_score > best_metric
        best_metric = max(final_score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.EXP,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric,
                     "optimizer": optimizer.state_dict()}
        if cycle is not None:
            save_dict["cycle"] = cycle
            save_filename = f"{cfg.EXP}_cycle{cycle}.pth"
        else:
            save_filename = f"{cfg.EXP}.pth"
        save_checkpoint(save_dict, is_best,
                        root=cfg.DIRS.WEIGHTS, filename=save_filename)
        return best_metric


def test_model(_print, cfg, model, test_loader):
    # TODO
    pass


def train_loop(_print, cfg, model, train_loader,
               criterion, optimizer, scheduler, epoch):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    use_cutmix = cfg.DATA.CUTMIX
    use_mixup = cfg.DATA.MIXUP
    for i, (image, mask) in enumerate(tbar):
        image = image.cuda()
        mask = mask.cuda()
        # mixup/ cutmix
        if use_mixup:
            image, mask = mixup_data(image, mask, alpha=cfg.DATA.CM_ALPHA)
        elif use_cutmix:
            image, mask = cutmix_data(image, mask, alpha=cfg.DATA.CM_ALPHA)
        # compute loss
        output = model(image)
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            output = torch.flatten(output, 1, -1)
            mask = torch.flatten(mask, 1, -1)
        loss = criterion(output, mask)
        # loss = binary_dice_loss(output, mask) + binary_iou_loss(output, mask)
        # gradient accumulation
        loss = loss / cfg.OPT.GD_STEPS
        if cfg.SYSTEM.FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.OPT.GD_STEPS == 0:
            scheduler(optimizer, i, epoch)
            optimizer.step()
            optimizer.zero_grad()
        # record loss
        losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


def distil_train_loop(_print, cfg, student_model, teacher_model,
                      train_loader, criterion, optimizer, scheduler, epoch):
    # TODO
    # _print(f"\nEpoch {epoch + 1}")

    # losses = AverageMeter()
    # student_model.train()
    # teacher_model.eval()
    # tbar = tqdm(train_loader)
    pass


def compute_jsd_loss(logits_clean, logits_aug1, logits_aug2, lamb=12.):
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean,
                                        dim=1), F.softmax(logits_aug1,
                                                          dim=1), F.softmax(logits_aug2,
                                                                            dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1.).log()
    jsd = lamb * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                  F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3.
    return jsd


def compute_distil_loss(student_logit, teacher_logit, temp=4.):
    student_prob = F.softmax(student_logit / temp, dim=-1)
    teacher_prob = F.softmax(teacher_logit / temp, dim=-1).log()
    loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
    return loss


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _, _, _, _) in enumerate(tbar):
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))