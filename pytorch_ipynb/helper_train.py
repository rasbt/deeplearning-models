from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss

import time
import torch
import torch.nn.functional as F


def train_classifier_simple_v1(num_epochs, model, optimizer, device, 
                               train_loader, valid_loader=None, 
                               loss_fn=None, logging_interval=100, 
                               skip_epoch_stats=False):

    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            cost = loss_fn(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), cost))

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference
                print('Epoch: %03d/%03d | Train Acc.: %.3f%% |  Loss: %.3f' % (
                      epoch+1, num_epochs,
                      compute_accuracy(model, train_loader, device),
                      compute_epoch_loss(model, train_loader, device)))

                if valid_loader is not None:
                    print('Epoch: %03d/%03d | Validation Acc.: %.3f%% |  Loss: %.3f' % (
                        epoch+1, num_epochs,
                        compute_accuracy(model, valid_loader, device),
                        compute_epoch_loss(model, valid_loader, device)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))