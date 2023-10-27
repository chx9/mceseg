def train_batch(net, X, y, loss, trainer, device):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
#     l.sum().backward()
    l.backward()
    trainer.step()
#     train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
#     return train_loss_sum, train_acc_sum
    return l, train_acc_sum


def train(net, train_iter, test_iter, loss, trainer, num_epochs,
          device=d2l.try_all_gpus(), scheduler=None):
    """Train a model with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    test_ious = torch.tensor(0, dtype=torch.float32).to(device)
    net = net.to(device)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(5)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, device)
            # loss_sum, accurant_count, batch_num, elements_count
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches % 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,  # x
                             # (metric[0] / metric[2], # loss
                             (metric[0] / (i+1),  # loss
                              metric[1] / metric[3],  # accurancy rate,
                              None))
        test_acc = evaluate_accuracy(net, test_iter)
        test_iou = evaluate_iou(net, test_iter)
        test_ious += test_iou
        animator.add(epoch, (None, None, test_acc))
        if scheduler:
            scheduler.step()
#     print(f'loss {metric[0] / metric[2]:.3f}, train acc '
    print(f'loss {metric[0] :.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(device)}')
    print(f'{test_ious.item()/num_epochs:.3f} test_iou')
